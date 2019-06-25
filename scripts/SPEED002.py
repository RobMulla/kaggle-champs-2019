'''
Created by: Rob Mulla
Jun 21, 2019

- Speed testing different approaches for training light gbm models
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
######################
## Helper Func
######################
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


#####################
# READ INPUT FILES
#####################
logger.info('Reading input files....')
path = 'data/'
train_df = pd.read_parquet(f'{path}/FE005-train.parquet')
test_df = pd.read_parquet(f'{path}/FE005-test.parquet')
ss = pd.read_csv('input/sample_submission.csv')
structures = pd.read_csv('input/structures.csv')
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)
train_df['atom_0'] = train_df['atom_0'].astype('category')
train_df['atom_1'] = train_df['atom_1'].astype('category')
test_df['atom_0'] = test_df['atom_0'].astype('category')
test_df['atom_1'] = test_df['atom_1'].astype('category')

train_df['type_0'] = train_df['type_0'].astype('category')
test_df['type_0'] = test_df['type_0'].astype('category')


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
MODEL_NUMBER = 'SPEED002'
# Make a runid that is unique to the time this is run for easy tracking later
run_id = "{:%m%d_%H%M}".format(datetime.now())
LEARNING_RATE = 0.1
FEATURES = ['bond_lengths_mean_y',
             'molecule_atom_index_0_dist_max',
             'bond_lengths_mean_x',
             'molecule_atom_index_0_dist_mean',
             'molecule_atom_index_0_dist_std',
             'molecule_couples',
             'molecule_atom_index_0_y_1_std',
             'molecule_dist_mean',
             'molecule_dist_max',
             'dist_y',
             'molecule_atom_index_0_z_1_std',
             'molecule_atom_index_1_dist_max',
             'molecule_atom_index_1_dist_min',
             'molecule_atom_index_0_x_1_std',
             'molecule_atom_index_1_dist_std',
             'molecule_atom_index_0_y_1_mean_div',
             'y_0',
             'molecule_atom_index_1_dist_mean',
             'molecule_atom_1_dist_mean',
             'x_0',
             'dist_x',
             'molecule_type_dist_std',
             'dist_z',
             'molecule_atom_index_1_dist_std_diff',
             'molecule_type_dist_mean_diff',
             'molecule_atom_index_0_dist_max_div',
             'molecule_atom_1_dist_std',
             'molecule_type_0_dist_std',
             'z_0',
             'molecule_type_dist_std_diff',
             'molecule_atom_index_0_y_1_mean_diff',
             'molecule_atom_index_0_dist_std_diff',
             'molecule_atom_index_0_dist_mean_div',
             'molecule_atom_index_0_dist_max_diff',
             'x_1',
             'molecule_type_dist_max',
             'molecule_atom_index_0_dist_std_div',
             'molecule_atom_index_0_dist_mean_diff',
             'molecule_atom_1_dist_std_diff',
             'molecule_atom_index_0_y_1_max_diff',
             'z_1',
             'molecule_atom_index_0_y_1_max',
             'molecule_atom_index_0_y_1_mean',
             'y_1',
             'molecule_type_0_dist_std_diff',
             'molecule_dist_min',
             'molecule_atom_index_1_dist_std_div',
             'molecule_atom_1_dist_min',
             'molecule_atom_index_1_dist_max_diff',
                            'type']
TARGET = 'scalar_coupling_constant'
CAT_FEATS = ['atom_0', 'atom_1']
N_ESTIMATORS = 20000
VERBOSE = 5000
EARLY_STOPPING_ROUNDS = 500
RANDOM_STATE = 529
N_THREADS = 42
n_fold = 3
EVAL_METRIC = 'group_mae'

update_tracking(run_id, 'model_number', MODEL_NUMBER)
update_tracking(run_id, 'n_estimators', N_ESTIMATORS)
update_tracking(run_id, 'early_stopping_rounds', EARLY_STOPPING_ROUNDS)
update_tracking(run_id, 'random_state', RANDOM_STATE)
update_tracking(run_id, 'n_threads', N_THREADS)
update_tracking(run_id, 'learning_rate', LEARNING_RATE)
update_tracking(run_id, 'n_fold', n_fold)
update_tracking(run_id, 'n_features', len(FEATURES))
update_tracking(run_id, 'model_type', 'lgbm')
update_tracking(run_id, 'eval_metric', EVAL_METRIC)

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
logger.info('Using features {}'.format([x for x in X.columns]))
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
              'random_state':RANDOM_STATE
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
for bond_type in ['2JHC']:
    
    ##############################
    #### TEST 1 ##################
    ##############################
    N_ESTIMATORS = 5000
    N_THREADS = 42
    bond_start = timer()
    fold_count = 1
    # Train the model
    X_type = X.loc[X['type'] == bond_type]
    y_type = y.iloc[X_type.index]
    X_test_type = X_test.loc[X_test['type'] == bond_type]
    oof = np.zeros(len(X_type))
    prediction_type = np.zeros(len(X_test_type))
    bond_scores = []
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X_type)):
        fold_start = timer()
        logger.info('Running Type {} - Fold {} of {}'.format(bond_type,
                                                       fold_count, folds.n_splits))
        X_train, X_valid = X_type.iloc[train_idx], X_type.iloc[valid_idx]
        y_train, y_valid = y_type.iloc[train_idx], y_type.iloc[valid_idx]
        logger.info(X_train.dtypes)
        train_ds = lgb.Dataset(X_train.drop('type', axis=1), label=y_train, free_raw_data=False)
        valid_ds = lgb.Dataset(X_valid.drop('type', axis=1), label=y_valid, free_raw_data=False)
        test_ds = lgb.Dataset(X_test_type.drop('type', axis=1), free_raw_data=False)
        lgb_params_temp = lgb_params.copy()
        lgb_params_temp['n_jobs'] = N_THREADS
        model = lgb.train(lgb_params_temp, 
                          train_ds,
                          num_boost_round=N_ESTIMATORS,
                          valid_sets=valid_ds,
                          feature_name=X_train.drop('type', axis=1).columns.tolist(),
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          verbose_eval=VERBOSE
                         )
        logger.info('Saving model file')
        model.save_model('models/{}-{}-{}-{}.model'.format(MODEL_NUMBER,
                                                           run_id,
                                                           bond_type,
                                                           fold_count))
        logger.info('Predicting on validation set')
        y_pred_valid = model.predict(X_valid.drop('type', axis=1))
        logger.info('Predicting on test set')
        y_pred = model.predict(X_test_type.drop('type', axis=1))

        # feature importance
        logger.info('Storing the fold importance')
        bond_scores.append(mean_absolute_error(y_valid, y_pred_valid))
        logger.info('CV mean score: {0:.4f}, std: {1:.4f}.'.format(
            np.mean(bond_scores), np.std(bond_scores)))
        oof[valid_idx] = y_pred_valid.reshape(-1,)
        prediction_type += y_pred
        fold_count += 1
        now = timer()
        logger.info('Completed training and predicting for bond {} fold {}-of-{} in {:0.4f} seconds'.format(bond_type,
                                                                                                            fold_n+1,
                                                                                                            fold_count,
                                                                                                            now-fold_start))
        break
    
    ##############################
    #### TEST 2 ##################
    ##############################
    N_ESTIMATORS = 5000
    N_THREADS = 42
    bond_start = timer()
    fold_count = 1
    # Train the model
    X_type = X.loc[X['type'] == bond_type]
    y_type = y.iloc[X_type.index]
    X_test_type = X_test.loc[X_test['type'] == bond_type]
    oof = np.zeros(len(X_type))
    prediction_type = np.zeros(len(X_test_type))
    bond_scores = []
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X_type)):
        fold_start = timer()
        logger.info('Running Type {} - Fold {} of {}'.format(bond_type,
                                                       fold_count, folds.n_splits))
        X_train, X_valid = X_type.iloc[train_idx], X_type.iloc[valid_idx]
        y_train, y_valid = y_type.iloc[train_idx], y_type.iloc[valid_idx]
        model = lgb.LGBMRegressor(**lgb_params, n_estimators=N_ESTIMATORS, n_jobs=N_THREADS)
        model.fit(X_train.drop('type', axis=1), y_train,
                  eval_set=[#(X_train.drop('type', axis=1), y_train),
                            (X_valid.drop('type', axis=1), y_valid)],
                  eval_metric=EVAL_METRIC,
                  verbose=VERBOSE,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        logger.info('Saving model file')
        model.booster_.save_model('models/{}-{}-{}-{}.model'.format(MODEL_NUMBER,
                                                           run_id,
                                                           bond_type,
                                                           fold_count))
        logger.info('Predicting on validation set')
        y_pred_valid = model.predict(X_valid.drop('type', axis=1),
                                     num_iteration=model.best_iteration_)
        logger.info('Predicting on test set')
        y_pred = model.predict(X_test_type.drop('type', axis=1),
                               num_iteration=model.best_iteration_)

        # feature importance
        logger.info('Storing the fold importance')
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
        now = timer()
        logger.info('Completed training and predicting for bond {} fold {}-of-{} in {:0.4f} seconds'.format(bond_type,
                                                                                                            fold_n+1,
                                                                                                            fold_count,
                                                                                                            now-fold_start))
        break
    ##############################
    #### TEST 3 ##################
    ##############################
    N_ESTIMATORS = 5000
    N_THREADS = 16
    bond_start = timer()
    fold_count = 1
    # Train the model
    X_type = X.loc[X['type'] == bond_type]
    y_type = y.iloc[X_type.index]
    X_test_type = X_test.loc[X_test['type'] == bond_type]
    oof = np.zeros(len(X_type))
    prediction_type = np.zeros(len(X_test_type))
    bond_scores = []
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X_type)):
        fold_start = timer()
        logger.info('Running Type {} - Fold {} of {}'.format(bond_type,
                                                       fold_count, folds.n_splits))
        X_train, X_valid = X_type.iloc[train_idx], X_type.iloc[valid_idx]
        y_train, y_valid = y_type.iloc[train_idx], y_type.iloc[valid_idx]
        model = lgb.LGBMRegressor(**lgb_params, n_estimators=N_ESTIMATORS, n_jobs=N_THREADS)
        model.fit(X_train.drop('type', axis=1), y_train,
                  eval_set=[#(X_train.drop('type', axis=1), y_train),
                            (X_valid.drop('type', axis=1), y_valid)],
                  eval_metric=EVAL_METRIC,
                  verbose=VERBOSE,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        logger.info('Saving model file')
        model.booster_.save_model('models/{}-{}-{}-{}.model'.format(MODEL_NUMBER,
                                                           run_id,
                                                           bond_type,
                                                           fold_count))
        logger.info('Predicting on validation set')
        y_pred_valid = model.predict(X_valid.drop('type', axis=1),
                                     num_iteration=model.best_iteration_)
        logger.info('Predicting on test set')
        y_pred = model.predict(X_test_type.drop('type', axis=1),
                               num_iteration=model.best_iteration_)

        # feature importance
        logger.info('Storing the fold importance')
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
        now = timer()
        logger.info('Completed training and predicting for bond {} fold {}-of-{} in {:0.4f} seconds'.format(bond_type,
                                                                                                            fold_n+1,
                                                                                                            fold_count,
                                                                                                            now-fold_start))
        break
    ##############################
    #### TEST 4 ##################
    ##############################
    N_ESTIMATORS = 5000
    N_THREADS = 8
    bond_start = timer()
    fold_count = 1
    # Train the model
    X_type = X.loc[X['type'] == bond_type]
    y_type = y.iloc[X_type.index]
    X_test_type = X_test.loc[X_test['type'] == bond_type]
    oof = np.zeros(len(X_type))
    prediction_type = np.zeros(len(X_test_type))
    bond_scores = []
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X_type)):
        fold_start = timer()
        logger.info('Running Type {} - Fold {} of {}'.format(bond_type,
                                                       fold_count, folds.n_splits))
        X_train, X_valid = X_type.iloc[train_idx], X_type.iloc[valid_idx]
        y_train, y_valid = y_type.iloc[train_idx], y_type.iloc[valid_idx]
        model = lgb.LGBMRegressor(**lgb_params, n_estimators=N_ESTIMATORS, n_jobs=N_THREADS)
        model.fit(X_train.drop('type', axis=1), y_train,
                  eval_set=[#(X_train.drop('type', axis=1), y_train),
                            (X_valid.drop('type', axis=1), y_valid)],
                  eval_metric=EVAL_METRIC,
                  verbose=VERBOSE,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        logger.info('Saving model file')
        model.booster_.save_model('models/{}-{}-{}-{}.model'.format(MODEL_NUMBER,
                                                           run_id,
                                                           bond_type,
                                                           fold_count))
        logger.info('Predicting on validation set')
        y_pred_valid = model.predict(X_valid.drop('type', axis=1),
                                     num_iteration=model.best_iteration_)
        logger.info('Predicting on test set')
        y_pred = model.predict(X_test_type.drop('type', axis=1),
                               num_iteration=model.best_iteration_)

        # feature importance
        logger.info('Storing the fold importance')
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
        now = timer()
        logger.info('Completed training and predicting for bond {} fold {}-of-{} in {:0.4f} seconds'.format(bond_type,
                                                                                                            fold_n+1,
                                                                                                            fold_count,
                                                                                                            now-fold_start))
        break