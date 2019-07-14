'''
Created by: Rob Mulla
Jun 10 8:14AM

TRACK THE SPEED VS THREAD COUNT
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
MODEL_NUMBER = 'SPEED001'
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
N_ESTIMATORS = 1000
VERBOSE = 5000
EARLY_STOPPING_ROUNDS = 500
RANDOM_STATE = 529
#N_THREADS = 32
n_fold = 5

update_tracking(run_id, 'model_number', MODEL_NUMBER)
update_tracking(run_id, 'n_estimators', N_ESTIMATORS)
update_tracking(run_id, 'early_stopping_rounds', EARLY_STOPPING_ROUNDS)
update_tracking(run_id, 'random_state', RANDOM_STATE)
#update_tracking(run_id, 'n_threads', N_THREADS)
update_tracking(run_id, 'learning_rate', LEARNING_RATE)
update_tracking(run_id, 'n_fold', n_fold)
update_tracking(run_id, 'n_features', len(FEATURES))

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
lgb_params = {'num_leaves': 128,
              'min_child_samples': 64,
              'objective': 'regression',
              'max_depth': 6,
              'learning_rate': LEARNING_RATE,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.9,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.4,
              'colsample_bytree': 1.0,
              }

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
for N_THREADS in [20, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]:
    start = timer()
    bond_type = X['type'].unique()[0]
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
        model = lgb.LGBMRegressor(**lgb_params, n_estimators=N_ESTIMATORS, n_jobs=N_THREADS)
        model.fit(X_train.drop('type', axis=1), y_train,
                  eval_set=[(X_train.drop('type', axis=1), y_train),
                            (X_valid.drop('type', axis=1), y_valid)],
                  eval_metric='mae',
                  verbose=VERBOSE,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.booster_.save_model('models/{}-{}-{}-{}.model'.format(MODEL_NUMBER,
                                                           run_id,
                                                           bond_type,
                                                           fold_count))
        y_pred_valid = model.predict(X_valid.drop('type', axis=1),
                                     num_iteration=model.best_iteration_)
        y_pred = model.predict(X_test_type.drop('type', axis=1),
                               num_iteration=model.best_iteration_)

        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = X_train.drop('type', axis=1).columns
        fold_importance["importance"] = model.feature_importances_
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat(
            [feature_importance, fold_importance], axis=0)

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
    end = timer()
    logger.info('====FOR THREADS {} - TIME IS {} seconds ======'.format(N_THREADS, (end - start)))
    bond_count += 1


logger.info('Done!')
