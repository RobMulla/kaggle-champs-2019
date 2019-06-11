'''
Created by: Rob Mulla
Jun 10 8:14AM

Changes:
    - Adding Tracking sheet functionality
    - Adding fixing feature importance CSV to include feature names
    - Save model files
    - Only 32 cores
'''
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import warnings
from datetime import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import gc
from timeit import default_timer as timer
import catboost
from catboost import CatBoostRegressor, Pool

start = timer()

#####################
## SETUP LOGGER
#####################
def get_logger():
    '''
        credits to: https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    '''
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
logger = get_logger()
#####################
# READ INPUT FILES
#####################
logger.info('Reading input files....')
train_df = pd.read_csv('data/FE004-train.csv')
test_df = pd.read_csv('data/FE004-test.csv')
ss = pd.read_csv('input/sample_submission.csv')
structures = pd.read_csv('input/structures.csv')

train_df['atom_0'] = train_df['atom_0'].astype('category')
train_df['atom_1'] = train_df['atom_1'].astype('category')
test_df['atom_0'] = test_df['atom_0'].astype('category')
test_df['atom_1'] = test_df['atom_1'].astype('category')
#####################
# FEATURE CREATION
#####################
logger.info('Creating features....')
logger.info('Available features {}'.format([x for x in train_df.columns]))
##########################
# Tracking Sheet function
#########################
def update_tracking(run_id, field, value, csv_file='tracking/tracking.csv'):
    df = pd.read_csv(csv_file, index_col=[0])
    df.loc[run_id, field] = value # Model number is index
    df.to_csv(csv_file)

####################
# CONFIGURABLES
#####################
# MODEL NUMBER
MODEL_NUMBER = 'M013'
# Make a runid that is unique to the time this is run for easy tracking later
run_id = "{:%m%d_%H%M}".format(datetime.now())
LEARNING_RATE = 0.5
FEATURES = ['type',
            'molecule_atom_index_0_dist_std_div', 'atom_index_0',
            'molecule_atom_index_1_dist_max_diff',
            'molecule_atom_index_1_dist_std_div',
            'molecule_atom_index_0_dist_max_diff', 'molecule_dist_mean', 'dist',
            'molecule_dist_max', 'molecule_atom_index_1_dist_std_diff',
            'molecule_atom_index_0_dist_std_diff', 'molecule_atom_index_1_dist_max',
            'molecule_atom_index_0_dist_std',
            'molecule_atom_index_0_dist_mean_diff',
            'molecule_atom_index_0_dist_max', 'molecule_couples',
            'molecule_atom_index_0_dist_mean_div',
            'molecule_atom_index_1_dist_max_div',
            'molecule_atom_index_0_dist_max_div',
            'molecule_atom_index_1_dist_mean_diff',
            'molecule_atom_index_1_dist_std', 'molecule_atom_index_1_dist_min',
            'dist_to_type_std', 'molecule_atom_index_1_dist_mean',
            'atom_0_couples_count', 'molecule_atom_index_1_dist_mean_div',
            'atom_1_couples_count', 'dist_to_type_mean']
TARGET = 'scalar_coupling_constant'
CAT_FEATS = ['atom_0', 'atom_1']
N_ESTIMATORS = 50000
VERBOSE = 5000
EARLY_STOPPING_ROUNDS = 500
RANDOM_STATE = 529
N_THREADS = 32
n_fold = 5

update_tracking(run_id, 'model_number', MODEL_NUMBER)
update_tracking(run_id, 'n_estimators', N_ESTIMATORS)
update_tracking(run_id, 'early_stopping_rounds', EARLY_STOPPING_ROUNDS)
update_tracking(run_id, 'random_state', RANDOM_STATE)
update_tracking(run_id, 'n_threads', N_THREADS)
update_tracking(run_id, 'learning_rate', LEARNING_RATE)
update_tracking(run_id, 'n_fold', n_fold)
update_tracking(run_id, 'n_features', len(FEATURES))
update_tracking(run_id, 'model_type', 'catboost')

#####################
# CREATE FINAL DATASETS
#####################
X = train_df[FEATURES]
X_test = test_df[FEATURES]
y = train_df[TARGET]

#####################
# TRAIN MODEL
#####################
logger.info('Training model....')
logger.info('Using features {}'.format([x for x in FEATURES]))

folds = KFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE)

# Setup arrays for storing results
oof_df = train_df[['id', 'type', 'scalar_coupling_constant']].copy()
oof_df['oof_preds'] = 0
prediction = np.zeros(len(X_test))
feature_importance = pd.DataFrame()
test_pred_df = test_df.copy()
test_pred_df['prediction'] = 0
bond_count = 1
number_of_bonds = len(X['type'].unique())
for bond_type in X['type'].unique():
    fold_count = 1
    # Train the model
    X_type = X.loc[X['type'] == bond_type]
    y_type = y.iloc[X_type.index]
    X_test_type = X_test.loc[X_test['type'] == bond_type]
    oof = np.zeros(len(X_type))
    prediction_type = np.zeros(len(X_test_type))
    bond_scores = []
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X_type)):
        logger.info('Running Type {} - Fold {} of {}'.format(bond_type,
                                                       fold_count, folds.n_splits))
        X_train, X_valid = X_type.iloc[train_idx], X_type.iloc[valid_idx]
        y_train, y_valid = y_type.iloc[train_idx], y_type.iloc[valid_idx]
        train_dataset = Pool(data=X_train.drop('type', axis=1), label=y_train)
        valid_dataset = Pool(data=X_valid.drop('type', axis=1), label=y_valid)
        test_dataset = Pool(data=X_test_type.drop('type', axis=1))
        model = CatBoostRegressor(iterations=N_ESTIMATORS,
                                     learning_rate=LEARNING_RATE,
                                     depth=7,
                                     eval_metric='MAE',
                                     verbose=5000,
                                     random_state = RANDOM_STATE,
                                     thread_count=N_THREADS)

        model.fit(train_dataset,
                  eval_set=valid_dataset,
                  early_stopping_rounds=500)
        model.save_model('models/{}-{}-{}-{}.model'.format(MODEL_NUMBER,
                                                           run_id,
                                                           bond_type,
                                                           fold_count))
        y_pred_valid = model.predict(valid_dataset)
        y_pred = model.predict(test_dataset)

        bond_scores.append(mean_absolute_error(y_valid, y_pred_valid))
        logger.info('CV mean score: {0:.4f}, std: {1:.4f}.'.format(
            np.mean(bond_scores), np.std(bond_scores)))
        oof[valid_idx] = y_pred_valid.reshape(-1,)
        prediction_type += y_pred
        fold_count += 1
    update_tracking(run_id, f'{bond_type}_mae_cv', np.mean(bond_scores))
    update_tracking(run_id, f'{bond_type}_std_mae_cv', np.std(bond_scores))
    oof_df.loc[oof_df['type'] == bond_type, 'oof_preds'] = oof
    prediction_type /= folds.n_splits
    test_pred_df.loc[test_pred_df['type'] ==
                     bond_type, 'prediction'] = prediction_type

    if bond_count != number_of_bonds:
        # Save the results inbetween bond type because it takes a long time
        submission_csv_name = 'temp/temp{}of{}_{}_{}_submission_lgb_{}folds_{}iter_{}lr.csv'.format(
            bond_count, number_of_bonds, MODEL_NUMBER, run_id, n_fold, N_ESTIMATORS, LEARNING_RATE)
        oof_csv_name = 'temp/temp{}of{}_{}_{}_oof_lgb_{}folds_{}iter_{}lr.csv'.format(
            bond_count, number_of_bonds, MODEL_NUMBER, run_id, n_fold, N_ESTIMATORS, LEARNING_RATE)
        logger.info('Saving Temporary LGB Submission files:')
        logger.info(submission_csv_name)
        ss = pd.read_csv('input/sample_submission.csv')
        ss['scalar_coupling_constant'] = test_pred_df['prediction']
        ss.to_csv(submission_csv_name, index=False)
        ss.head()
        # OOF
        oof_df.to_csv(oof_csv_name, index=False)
    bond_count += 1
oof_score = mean_absolute_error(
    oof_df['scalar_coupling_constant'], oof_df['oof_preds'])
update_tracking(run_id, 'oof_score', oof_score)
logger.info('Out of fold score is {:.4f}'.format(oof_score))

#####################
# SAVE RESULTS
#####################
# Save Prediction and name appropriately
submission_csv_name = 'submissions/{}_{}_submission_lgb_{}folds_{:.4f}CV_{}iter_{}lr.csv'.format(MODEL_NUMBER,
                                                                                                 run_id,
                                                                                                 n_fold,
                                                                                                 oof_score,
                                                                                                 N_ESTIMATORS,
                                                                                                 LEARNING_RATE)
oof_csv_name = 'oof/{}_{}_oof_lgb_{}folds_{:.4f}CV_{}iter_{}lr.csv'.format(
    MODEL_NUMBER, run_id, n_fold, oof_score, N_ESTIMATORS, LEARNING_RATE)

logger.info('Saving LGB Submission as:')
logger.info(submission_csv_name)
ss = pd.read_csv('input/sample_submission.csv')
ss['scalar_coupling_constant'] = test_pred_df['prediction']
ss.to_csv(submission_csv_name, index=False)
ss.head()
# OOF
oof_df.to_csv(oof_csv_name, index=False)

# Finish
end = timer()
update_tracking(run_id, 'training_time', (end-start))
logger.info('==== Training done in {} seconds ======'.format(end - start))
logger.info('Done!')
