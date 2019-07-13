'''
Created by: Rob Mulla
Jul 12
New Changes:
    - Calculate meta feature for FC
    - Calculated within fold
Old Changes:
    - Features from FE018
    - Features from FE017
    - Remove features per type if feature is all nulls
    - change logging timestamp
    - update code to check for model number being same as filename
    - FE010 !
    - Load feature data each fold
    - N_THREADS when using predict
    - Remove useless features
    - Delete and gc train and test df after copied
    - Fixed OOF error by using GroupKFold
    - Adding type column to feature importance dataframe/csv
    - Tracking sheet set percision
    - New features created from openbabel
    - Switch to GroupShuffleSplit
    - CV by molecule type. Reduce overfitting of CV score
    - Caclulate group mean log mae score also
'''
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import warnings
from datetime import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import gc
from timeit import default_timer as timer
import time
from catboost import CatBoostRegressor, Pool
from sklearn.neighbors import KNeighborsClassifier
start = timer()

#####################
## SETUP LOGGER
#####################
def get_logger():
    '''
        credits to: https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    '''
    os.environ['TZ'] = 'US/Eastern'
    time.tzset()
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
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

def group_mean_log_mae(y_true, y_pred, groups, floor=1e-9):
    maes = (y_true-y_pred).abs().groupby(groups).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()

#####################
# READ INPUT FILES
#####################
logger.info('Reading input files....')
path = 'data/'
# train_df = pd.read_parquet(f'{path}FE009_train_pandas.parquet')
# test_df = pd.read_parquet(f'{path}FE009_test_pandas.parquet')
ss = pd.read_csv('input/sample_submission.csv')

#####################
# FEATURE CREATION
#####################
# logger.info('Creating features....')
# logger.info('Available features {}'.format([x for x in train_df.columns]))
##########################
# Tracking Sheet function
#########################
def update_tracking(run_id, field, value, csv_file='tracking/tracking.csv',
                    integer=False, digits=None):
    df = pd.read_csv(csv_file, index_col=[0])
    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)
    df.loc[run_id, field] = value # Model number is index
    df.to_csv(csv_file)

####################
# CONFIGURABLES
#####################

FEATURES = [
            '10th_closest_to_0_atomic_mass_x_cube_inv_dist',
             '10th_closest_to_0_valence_x_cube_inv_dist',
             '10th_closest_to_1_atomic_mass_x_cube_inv_dist',
             '10th_closest_to_1_valence_x_cube_inv_dist',
             '12th_closest_to_0_atomic_mass_x_cube_inv_dist',
             '12th_closest_to_1_atomic_mass_x_cube_inv_dist',
             '13th_closest_to_1_atomic_mass_x_cube_inv_dist',
             '14th_closest_to_1_atomic_mass_x_cube_inv_dist',
             '21st_closest_to_0_atomic_mass_x_cube_inv_dist',
             '2nd_closest_to_0_atomic_mass',
             '2nd_closest_to_0_atomic_mass_x_cube_inv_dist',
             '2nd_closest_to_0_dist_x_atomic_mass',
             '2nd_closest_to_0_valence',
             '2nd_closest_to_0_valence_x_cube_inv_dist',
             '2nd_closest_to_1_atomic_mass',
             '2nd_closest_to_1_atomic_mass_x_cube_inv_dist',
             '2nd_closest_to_1_dist_x_atomic_mass',
             '2nd_closest_to_1_valence',
             '2nd_closest_to_1_valence_x_cube_inv_dist',
             '3rd_closest_to_0_atomic_mass',
             '3rd_closest_to_0_atomic_mass_x_cube_inv_dist',
             '3rd_closest_to_0_dist_x_atomic_mass',
             '3rd_closest_to_0_valence',
             '3rd_closest_to_0_valence_x_cube_inv_dist',
             '3rd_closest_to_1_atomic_mass',
             '3rd_closest_to_1_atomic_mass_x_cube_inv_dist',
             '3rd_closest_to_1_dist_x_atomic_mass',
             '3rd_closest_to_1_valence',
             '3rd_closest_to_1_valence_x_cube_inv_dist',
             '4th_closest_to_0_atomic_mass',
             '4th_closest_to_0_atomic_mass_x_cube_inv_dist',
             '4th_closest_to_0_dist_x_atomic_mass',
             '4th_closest_to_0_valence',
             '4th_closest_to_0_valence_x_cube_inv_dist',
             '4th_closest_to_1_atomic_mass_x_cube_inv_dist',
             '4th_closest_to_1_dist_x_atomic_mass',
             '4th_closest_to_1_valence',
             '4th_closest_to_1_valence_x_cube_inv_dist',
             '5th_closest_to_0_atomic_mass',
             '5th_closest_to_0_atomic_mass_x_cube_inv_dist',
             '5th_closest_to_0_dist_x_atomic_mass',
             '5th_closest_to_0_valence',
             '5th_closest_to_0_valence_x_cube_inv_dist',
             '5th_closest_to_1_atomic_mass',
             '5th_closest_to_1_atomic_mass_x_cube_inv_dist',
             '5th_closest_to_1_dist_x_atomic_mass',
             '5th_closest_to_1_valence',
             '5th_closest_to_1_valence_x_cube_inv_dist',
             '6th_closest_to_0_atomic_mass',
             '6th_closest_to_0_atomic_mass_x_cube_inv_dist',
             '6th_closest_to_0_dist_x_atomic_mass',
             '6th_closest_to_0_valence',
             '6th_closest_to_0_valence_x_cube_inv_dist',
             '6th_closest_to_1_atomic_mass',
             '6th_closest_to_1_atomic_mass_x_cube_inv_dist',
             '6th_closest_to_1_dist_x_atomic_mass',
             '6th_closest_to_1_valence_x_cube_inv_dist',
             '7th_closest_to_0_atomic_mass',
             '7th_closest_to_0_atomic_mass_x_cube_inv_dist',
             '7th_closest_to_0_dist_x_atomic_mass',
             '7th_closest_to_0_valence_x_cube_inv_dist',
             '7th_closest_to_1_atomic_mass_x_cube_inv_dist',
             '7th_closest_to_1_dist_x_atomic_mass',
             '7th_closest_to_1_valence',
             '7th_closest_to_1_valence_x_cube_inv_dist',
             '8th_closest_to_0_atomic_mass_x_cube_inv_dist',
             '8th_closest_to_0_dist_x_atomic_mass',
             '8th_closest_to_0_valence_x_cube_inv_dist',
             '8th_closest_to_1_atomic_mass_x_cube_inv_dist',
             '8th_closest_to_1_dist_x_atomic_mass',
             '8th_closest_to_1_valence_x_cube_inv_dist',
             '9th_closest_to_0_atomic_mass',
             '9th_closest_to_0_atomic_mass_x_cube_inv_dist',
             '9th_closest_to_0_dist_x_atomic_mass',
             '9th_closest_to_0_valence_x_cube_inv_dist',
             '9th_closest_to_1_atomic_mass_x_cube_inv_dist',
             '9th_closest_to_1_dist_x_atomic_mass',
             '9th_closest_to_1_valence_x_cube_inv_dist',
             'N1',
             'N2',
             'NH',
             'adC1',
             'adC2',
             'adC3',
             'adC4',
             'adH1',
             'adH2',
             'adN1',
             'adN2',
             'angle_0_10th0_1',
             'angle_0_10th1_1',
             'angle_0_1_2nd1',
             'angle_0_1_3rd1',
             'angle_0_1_4th1',
             'angle_0_1_5th1',
             'angle_0_1_closest1',
             'angle_0_2nd0_1',
             'angle_0_3rd0_1',
             'angle_0_3rd1_1',
             'angle_0_4th0_1',
             'angle_0_4th1_1',
             'angle_0_5th0_1',
             'angle_0_5th1_1',
             'angle_0_6th0_1',
             'angle_0_6th1_1',
             'angle_0_7th0_1',
             'angle_0_7th1_1',
             'angle_0_8th0_1',
             'angle_0_8th1_1',
             'angle_0_9th0_1',
             'angle_0_9th1_1',
             'angle_0_closest1_1',
             'angle_1_0_10th0',
             'angle_1_0_10th1',
             'angle_1_0_11th0',
             'angle_1_0_11th1',
             'angle_1_0_2nd0',
             'angle_1_0_3rd0',
             'angle_1_0_4th0',
             'angle_1_0_5th0',
             'angle_1_0_6th0',
             'angle_1_0_6th1',
             'angle_1_0_7th0',
             'angle_1_0_7th1',
             'angle_1_0_8th0',
             'angle_1_0_8th1',
             'angle_1_0_9th0',
             'angle_1_0_9th1',
             'angle_clos_0_2nd',
             'angle_clos_1_2nd',
             'atom1_valence',
             'atomic_mass_not_0_max',
             'atomic_mass_not_1_max',
             'atomic_mass_not_1_mean',
             'atomic_mass_not_1_std',
             'closest_to_0_atomic_mass_x_cube_inv_dist',
             'closest_to_0_dist_x_atomic_mass',
             'closest_to_0_spin_multiplicity',
             'closest_to_0_valence',
             'closest_to_0_valence_x_cube_inv_dist',
             'closest_to_1_atomic_mass_x_cube_inv_dist',
             'closest_to_1_dist_x_atomic_mass',
             'closest_to_1_valence_x_cube_inv_dist',
             'coulomb_H.x',
             'coulomb_H.y',
             'distC0',
             'distC1',
             'dist_to_0_max',
             'dist_to_0_min',
             'dist_to_1_max',
             'dist_to_1_min',
             'distance',
             'distance_10th_closest_to_0',
             'distance_10th_closest_to_1',
             'distance_10th_closest_to_1_cube_inverse',
             'distance_2nd_closest_to_0',
             'distance_2nd_closest_to_0_cube_inverse',
             'distance_2nd_closest_to_1',
             'distance_2nd_closest_to_1_cube_inverse',
             'distance_3rd_closest_to_0',
             'distance_3rd_closest_to_0_cube_inverse',
             'distance_3rd_closest_to_1',
             'distance_3rd_closest_to_1_cube_inverse',
             'distance_4th_closest_to_0',
             'distance_4th_closest_to_0_cube_inverse',
             'distance_4th_closest_to_1',
             'distance_4th_closest_to_1_cube_inverse',
             'distance_5th_closest_to_0',
             'distance_5th_closest_to_0_cube_inverse',
             'distance_5th_closest_to_1',
             'distance_5th_closest_to_1_cube_inverse',
             'distance_6th_closest_to_0',
             'distance_6th_closest_to_0_cube_inverse',
             'distance_6th_closest_to_1',
             'distance_6th_closest_to_1_cube_inverse',
             'distance_7th_closest_to_1',
             'distance_7th_closest_to_1_cube_inverse',
             'distance_8th_closest_to_1',
             'distance_8th_closest_to_1_cube_inverse',
             'distance_9th_closest_to_0',
             'distance_9th_closest_to_1',
             'distance_9th_closest_to_1_cube_inverse',
             'distance_closest_to_0',
             'distance_closest_to_0_cube_inverse',
             'distance_closest_to_1',
             'distance_closest_to_1_cube_inverse',
             'feat_acsf_g2_C_[1, 2]_atom0',
             'feat_acsf_g2_C_[1, 2]_atom1',
             'feat_acsf_g2_C_[1, 6]_atom0',
             'feat_acsf_g2_C_[1, 6]_atom1',
             'feat_acsf_g2_F_[0.01, 6]_atom1',
             'feat_acsf_g2_H_[0.01, 2]_atom1',
             'feat_acsf_g2_H_[0.01, 6]_atom1',
             'feat_acsf_g2_H_[0.1, 2]_atom1',
             'feat_acsf_g2_H_[0.1, 6]_atom1',
             'feat_acsf_g2_H_[1, 2]_atom0',
             'feat_acsf_g2_H_[1, 2]_atom1',
             'feat_acsf_g2_H_[1, 6]_atom0',
             'feat_acsf_g2_H_[1, 6]_atom1',
             'feat_acsf_g2_N_[0.01, 2]_atom0',
             'feat_acsf_g2_N_[1, 2]_atom1',
             'feat_acsf_g2_N_[1, 6]_atom1',
             'feat_acsf_g2_O_[1, 2]_atom0',
             'feat_acsf_g2_O_[1, 2]_atom1',
             'feat_acsf_g2_O_[1, 6]_atom0',
             'feat_acsf_g2_O_[1, 6]_atom1',
             'feat_acsf_g4_C_C_[0.01, 4, -1]_atom0',
             'feat_acsf_g4_C_C_[0.01, 4, -1]_atom1',
             'feat_acsf_g4_C_C_[0.01, 4, 1]_atom0',
             'feat_acsf_g4_C_C_[0.01, 4, 1]_atom1',
             'feat_acsf_g4_C_C_[0.1, 4, -1]_atom0',
             'feat_acsf_g4_C_C_[0.1, 4, -1]_atom1',
             'feat_acsf_g4_C_C_[0.1, 4, 1]_atom1',
             'feat_acsf_g4_C_C_[1, 4, -1]_atom0',
             'feat_acsf_g4_C_C_[1, 4, -1]_atom1',
             'feat_acsf_g4_C_C_[1, 4, 1]_atom0',
             'feat_acsf_g4_C_C_[1, 4, 1]_atom1',
             'feat_acsf_g4_C_H_[0.1, 4, -1]_atom0',
             'feat_acsf_g4_C_H_[0.1, 4, -1]_atom1',
             'feat_acsf_g4_C_H_[0.1, 4, 1]_atom0',
             'feat_acsf_g4_C_H_[0.1, 4, 1]_atom1',
             'feat_acsf_g4_C_H_[1, 4, -1]_atom0',
             'feat_acsf_g4_C_H_[1, 4, -1]_atom1',
             'feat_acsf_g4_C_H_[1, 4, 1]_atom0',
             'feat_acsf_g4_C_H_[1, 4, 1]_atom1',
             'feat_acsf_g4_H_H_[0.01, 4, -1]_atom0',
             'feat_acsf_g4_H_H_[0.01, 4, -1]_atom1',
             'feat_acsf_g4_H_H_[0.1, 4, -1]_atom0',
             'feat_acsf_g4_H_H_[0.1, 4, -1]_atom1',
             'feat_acsf_g4_H_H_[0.1, 4, 1]_atom0',
             'feat_acsf_g4_H_H_[0.1, 4, 1]_atom1',
             'feat_acsf_g4_H_H_[1, 4, -1]_atom0',
             'feat_acsf_g4_H_H_[1, 4, -1]_atom1',
             'feat_acsf_g4_H_H_[1, 4, 1]_atom0',
             'feat_acsf_g4_H_H_[1, 4, 1]_atom1',
             'feat_acsf_g4_N_C_[0.01, 4, -1]_atom0',
             'feat_acsf_g4_N_C_[0.1, 4, -1]_atom0',
             'feat_acsf_g4_N_C_[0.1, 4, -1]_atom1',
             'feat_acsf_g4_N_C_[0.1, 4, 1]_atom0',
             'feat_acsf_g4_N_C_[1, 4, -1]_atom0',
             'feat_acsf_g4_N_C_[1, 4, -1]_atom1',
             'feat_acsf_g4_N_C_[1, 4, 1]_atom0',
             'feat_acsf_g4_N_C_[1, 4, 1]_atom1',
             'feat_acsf_g4_N_H_[0.1, 4, 1]_atom1',
             'feat_acsf_g4_N_H_[1, 4, -1]_atom0',
             'feat_acsf_g4_N_H_[1, 4, -1]_atom1',
             'feat_acsf_g4_N_H_[1, 4, 1]_atom0',
             'feat_acsf_g4_N_H_[1, 4, 1]_atom1',
             'feat_acsf_g4_N_N_[0.01, 4, 1]_atom0',
             'feat_acsf_g4_N_N_[1, 4, -1]_atom1',
             'feat_acsf_g4_O_C_[0.01, 4, -1]_atom0',
             'feat_acsf_g4_O_C_[0.01, 4, -1]_atom1',
             'feat_acsf_g4_O_C_[0.01, 4, 1]_atom0',
             'feat_acsf_g4_O_C_[0.1, 4, -1]_atom0',
             'feat_acsf_g4_O_C_[0.1, 4, -1]_atom1',
             'feat_acsf_g4_O_C_[0.1, 4, 1]_atom1',
             'feat_acsf_g4_O_C_[1, 4, -1]_atom0',
             'feat_acsf_g4_O_C_[1, 4, -1]_atom1',
             'feat_acsf_g4_O_C_[1, 4, 1]_atom0',
             'feat_acsf_g4_O_C_[1, 4, 1]_atom1',
             'feat_acsf_g4_O_H_[0.01, 4, -1]_atom1',
             'feat_acsf_g4_O_H_[0.1, 4, -1]_atom0',
             'feat_acsf_g4_O_H_[0.1, 4, -1]_atom1',
             'feat_acsf_g4_O_H_[0.1, 4, 1]_atom0',
             'feat_acsf_g4_O_H_[0.1, 4, 1]_atom1',
             'feat_acsf_g4_O_H_[1, 4, -1]_atom0',
             'feat_acsf_g4_O_H_[1, 4, -1]_atom1',
             'feat_acsf_g4_O_H_[1, 4, 1]_atom0',
             'feat_acsf_g4_O_H_[1, 4, 1]_atom1',
             'feat_acsf_g4_O_N_[0.1, 4, -1]_atom1',
             'feat_acsf_g4_O_N_[0.1, 4, 1]_atom1',
             'feat_acsf_g4_O_N_[1, 4, -1]_atom0',
             'feat_acsf_g4_O_N_[1, 4, -1]_atom1',
             'feat_acsf_g4_O_N_[1, 4, 1]_atom0',
             'feat_acsf_g4_O_O_[0.1, 4, -1]_atom0',
             'feat_acsf_g4_O_O_[0.1, 4, -1]_atom1',
             'feat_acsf_g4_O_O_[1, 4, -1]_atom0',
             'feat_acsf_g4_O_O_[1, 4, -1]_atom1',
             'inv_dist0E',
             'inv_dist0R',
             'inv_dist1',
             'inv_dist1E',
             'inv_dist1R',
             'inv_distP',
             'inv_distPE',
             'inv_distPR',
             'linkM0',
             'linkM1',
             'linkN',
             'max_molecule_atom_0_dist_xyz',
             'max_molecule_atom_1_dist_xyz',
             'mean_molecule_atom_0_dist_xyz',
             'mean_molecule_atom_1_dist_xyz',
             'min_molecule_atom_0_dist_xyz',
             'min_molecule_atom_1_dist_xyz',
             'sd_molecule_atom_1_dist_xyz',
             'tor_ang_2leftleft_count',
             'tor_ang_2leftleft_max',
             'tor_ang_2leftleft_mean',
             'tor_ang_2leftleft_min',
             'val_not_0_mean',
             'val_not_0_min',
             'val_not_0_std',
             'val_not_1_mean',
             'val_not_1_std',
             'yukawa_C.x',
             'yukawa_H.x',
             'yukawa_H.y',
             'yukawa_N.x',
             'yukawa_O.x'
]

# MODEL NUMBER
MODEL_NUMBER = 'M046_3JHC'
script_name = os.path.basename(__file__).split('.')[0]
if script_name not in MODEL_NUMBER:
    logger.error('Model Number is not same as script! Update before running')
    raise SystemExit('Model Number is not same as script! Update before running')

# Make a runid that is unique to the time this is run for easy tracking later
run_id = "{:%m%d_%H%M}".format(datetime.now())
LEARNING_RATE = 0.1
RUN_SINGLE_FOLD = False # Fold number to run starting with 1 - Set to False to run all folds
TARGET = 'scalar_coupling_constant'
N_ESTIMATORS = 500000
N_META_ESTIMATORS = 50000
VERBOSE = 1000
EARLY_STOPPING_ROUNDS = 500
RANDOM_STATE = 529
N_THREADS = 48
N_FOLDS = 2
N_META_FOLDS = 2
#EVAL_METRIC = 'group_mae'
EVAL_METRIC = 'MAE'
MODEL_TYPE = 'catboost'
update_tracking(run_id, 'model_number', MODEL_NUMBER)
update_tracking(run_id, 'n_estimators', N_ESTIMATORS)
update_tracking(run_id, 'early_stopping_rounds', EARLY_STOPPING_ROUNDS)
update_tracking(run_id, 'random_state', RANDOM_STATE)
update_tracking(run_id, 'n_threads', N_THREADS)
update_tracking(run_id, 'learning_rate', LEARNING_RATE)
update_tracking(run_id, 'n_fold', N_FOLDS)
update_tracking(run_id, 'n_features', len(FEATURES))
update_tracking(run_id, 'model_type', MODEL_TYPE)
update_tracking(run_id, 'eval_metric', EVAL_METRIC)

#####################
# FUNCTION FOR META FEATURE CREATION
######################

META_DEPTH = 7

def fit_meta_feature(X_train, X_valid, X_test, Meta_train,
                     train_idx, bond_type, base_fold, feature='fc',
                     N_META_FOLDS=N_META_FOLDS,
                     N_META_ESTIMATORS=N_META_ESTIMATORS):
    """
    Adds meta features to train, test and val
    """
    logger.info(f'Creating meta feature {feature}')
    logger.info('X_train, X_valid and X_test are shapes {} {} {}'.format(X_train.shape,
                                                                         X_valid.shape,
                                                                         X_test.shape))
    folds = GroupKFold(n_splits=N_META_FOLDS)
    fold_count = 1
    
    # Init predictions
    X_valid['meta_'+feature] = 0
    X_test['meta_'+feature] = 0
    X_train['meta_'+feature] = 0
    X_train_oof = X_train[['meta_'+feature]].copy()
    X_train = X_train.drop('meta_'+feature, axis=1)
    
    for fold_n, (train_idx2, valid_idx2) in enumerate(folds.split(X_train, groups=mol_group_type.iloc[train_idx].values)):
        logger.info('Running Meta Feature Type {} - Fold {} of {}'.format(feature, fold_count, folds.n_splits))
        update_tracking(run_id, '{}_meta_{}_est'.format(bond_type, feature), N_META_ESTIMATORS)
        update_tracking(run_id, '{}_meta_{}_metafolds'.format(bond_type, feature), N_META_FOLDS)

        X_train2 = X_train.loc[X_train.reset_index().index.isin(train_idx2)]
        X_valid2 = X_train.loc[X_train.reset_index().index.isin(valid_idx2)]
        X_train2 = X_train2.copy()
        X_valid2 = X_valid2.copy()
        y_train2 = Meta_train.loc[Meta_train.reset_index().index.isin(train_idx2)][feature]
        y_valid2 = Meta_train.loc[Meta_train.reset_index().index.isin(valid_idx2)][feature]
        fold_count += 1

        train_dataset = Pool(data=X_train2, label=y_train2)
        metavalid_dataset = Pool(data=X_valid2, label=y_valid2)
        valid_dataset = Pool(data=X_valid)
        test_dataset = Pool(data=X_test)
        model = CatBoostRegressor(iterations=N_META_ESTIMATORS,
                                     learning_rate=LEARNING_RATE,
                                     depth=META_DEPTH,
                                     eval_metric=EVAL_METRIC,
                                     verbose=VERBOSE,
                                     random_state = RANDOM_STATE,
                                     thread_count=N_THREADS,
                                     # loss_function=EVAL_METRIC,
                                     # bootstrap_type='Poisson',
                                     # bagging_temperature=5,
                                     task_type = "GPU") # Train on GPU

        model.fit(train_dataset,
                  eval_set=metavalid_dataset,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        y_pred_meta_valid = model.predict(metavalid_dataset)
        y_pred_valid = model.predict(valid_dataset)
        y_pred = model.predict(test_dataset)
        
        X_train_oof.loc[X_train_oof.reset_index().index.isin(valid_idx2), 'meta_'+feature] = y_pred_meta_valid
        X_valid['meta_'+feature] += y_pred_valid
        X_test['meta_'+feature] += y_pred

    oof_score = mean_absolute_error(Meta_train[feature], X_train_oof['meta_'+feature])
    update_tracking(run_id, '{}_meta_{}_mae_cv_f{}'.format(bond_type, feature, base_fold), oof_score)

    X_valid['meta_'+feature] = X_valid['meta_'+feature] / N_META_FOLDS
    X_test['meta_'+feature] = X_test['meta_'+feature] / N_META_FOLDS
    X_train['meta_'+feature] = X_train_oof['meta_'+feature]
    logger.info('Done creating meta features')
    logger.info('X_train, X_valid and X_test are shapes {} {} {}'.format(X_train.shape, X_valid.shape, X_test.shape))
    return X_train, X_valid, X_test


#####################
# TRAIN MODEL
#####################
logger.info('Training model....')
logger.info('Using features {}'.format([x for x in FEATURES]))
lgb_params = {"boosting_type" : "gbdt",
              "objective" : "regression_l2",
              "learning_rate" : LEARNING_RATE,
              "num_leaves" : 255,
              "sub_feature" : 0.50,
              "sub_row" : 0.75,
              "bagging_freq" : 1,
              "metric" : EVAL_METRIC,
              'random_state': RANDOM_STATE
              }

folds = GroupKFold(n_splits=N_FOLDS)

# Setup arrays for storing results
train_df = pd.read_parquet('data/FE008_train.parquet') # only loading for skeleton not features
oof_df = train_df[['id', 'type','scalar_coupling_constant']].copy()
mol_group = train_df[['molecule_name','type']].copy()
del train_df
gc.collect()

oof_df['oof_preds'] = 0
test_df = pd.read_parquet('data/FE008_test.parquet') # only loading for skeleton not features
prediction = np.zeros(len(test_df))
feature_importance = pd.DataFrame()
test_pred_df = test_df[['id','type','molecule_name']].copy()
del test_df
gc.collect()
test_pred_df['prediction'] = 0
bond_count = 1

types = ['3JHC']
number_of_bonds = len(types)

#### CREATE FOLDERS FOR MODEL NUMBER IF THEY DONT EXIST

if not os.path.exists('models/{}'.format(MODEL_NUMBER)):
    os.makedirs('models/{}'.format(MODEL_NUMBER))
if not os.path.exists('temp/{}'.format(MODEL_NUMBER)):
    os.makedirs('temp/{}'.format(MODEL_NUMBER))

### Load Scalar Coupling Components
tr_scc = pd.read_parquet('data/tr_scc.parquet')
    
### RUN MODEL
for bond_type in types:
    # Read the files and make X, X_test, and y
    train_df = pd.read_parquet('data/FE018/FE018-train-{}.parquet'.format(bond_type))
    test_df = pd.read_parquet('data/FE018/FE018-test-{}.parquet'.format(bond_type))
    Meta = train_df[['molecule_name', 'atom_index_0', 'atom_index_1']].merge(
        tr_scc, on=['molecule_name', 'atom_index_0', 'atom_index_1'])[['fc','sd','pso','dso']]
    X_type = train_df[FEATURES].copy()
    X_test_type = test_df[FEATURES].copy()
    y_type = train_df[TARGET].copy()
    del train_df
    del test_df
    gc.collect()
    # Remove colmns that have all nulls
    logger.info('{} Features before dropping null columns'.format(len(X_type.columns)))
    X_type = X_type.dropna(how='all', axis=1)
    X_test_type = X_test_type[X_type.columns]
    logger.info('{} Features after dropping null columns'.format(len(X_type.columns)))
    # Start training for type
    bond_start = timer()
    fold_count = 0 # Will be incremented at the start of the fold
    # Train the model
    # X_type = X.loc[X['type'] == bond_type]
    # y_type = y.iloc[X_type.index]
    # X_test_type = X_test.loc[X_test['type'] == bond_type]
    mol_group_type = mol_group.loc[mol_group['type'] == bond_type]['molecule_name']
    oof = np.zeros(len(X_type))
    prediction_type = np.zeros(len(X_test_type))
    bond_scores = []
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X_type, groups=mol_group_type)):
        Meta_train, Meta_valid = Meta.iloc[train_idx], Meta.iloc[valid_idx]
        fold_count += 1 # First fold is 1
        if RUN_SINGLE_FOLD is not False:
            if fold_count != RUN_SINGLE_FOLD:
                logger.info('Running only for fold {}, skipping fold {}'.format(RUN_SINGLE_FOLD,
                                                                                fold_count))
                continue
        if MODEL_TYPE == 'lgbm':
            fold_start = timer()
            logger.info('Running Type {} - Fold {} of {}'.format(bond_type,
                                                           fold_count, folds.n_splits))
            X_train, X_valid = X_type.iloc[train_idx], X_type.iloc[valid_idx]
            y_train, y_valid = y_type.iloc[train_idx], y_type.iloc[valid_idx]
            X_train, X_valid, X_test_type = fit_meta_feature(X_train, X_valid,
                                                             X_test_type, Meta_train,
                                                             train_idx, bond_type, fold_count)
            model = lgb.LGBMRegressor(**lgb_params, n_estimators=N_ESTIMATORS, n_jobs=N_THREADS)
            model.fit(X_train, y_train,
                      eval_set=[#(X_train, y_train),
                                (X_valid, y_valid)],
                      eval_metric=EVAL_METRIC,
                      verbose=VERBOSE,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            now = timer()
            update_tracking(run_id, '{}_tr_sec_f{}'.format(bond_type, fold_n+1),
                            (now-fold_start), integer=True)
            logger.info('Saving model file')
            model.booster_.save_model('models/{}-{}-{}-{}.model'.format(MODEL_NUMBER,
                                                               run_id,
                                                               bond_type,
                                                               fold_count))
            pred_start = timer()
            logger.info('Predicting on validation set')
            y_pred_valid = model.predict(X_valid,
                                         num_iteration=model.best_iteration_,
                                         n_jobs=N_THREADS)
            logger.info('Predicting on test set')
            y_pred = model.predict(X_test_type,
                                   num_iteration=model.best_iteration_)
            now = timer()
            update_tracking(run_id, '{}_pred_sec_f{}'.format(bond_type, fold_n+1),
                            (now-pred_start), integer=True)
            update_tracking(run_id, '{}_f{}_best_iter'.format(bond_type, fold_n+1),
                            model.best_iteration_, integer=True)

            # feature importance
            logger.info('Storing the fold importance')
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X_train.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["type"] = bond_type
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0)

            bond_scores.append(mean_absolute_error(y_valid, y_pred_valid))
            logger.info('CV mean score: {0:.4f}, std: {1:.4f}.'.format(
                np.mean(bond_scores), np.std(bond_scores)))
            oof[valid_idx] = y_pred_valid.reshape(-1,)
            prediction_type += y_pred
        elif MODEL_TYPE == 'catboost':
            fold_start = timer()
            logger.info('Running Type {} - Fold {} of {}'.format(bond_type,
                                                           fold_count, folds.n_splits))
            X_train, X_valid = X_type.iloc[train_idx], X_type.iloc[valid_idx]
            X_train = X_train.copy()
            X_valid = X_valid.copy()
            y_train, y_valid = y_type.iloc[train_idx], y_type.iloc[valid_idx]

            ### ADD META FEATURES
            X_train, X_valid, X_test_type = fit_meta_feature(X_train, X_valid,
                                                             X_test_type, Meta_train,
                                                             train_idx, bond_type,
                                                             fold_count, feature='fc')
#             X_train, X_valid, X_test_type = fit_meta_feature(X_train, X_valid, X_test_type,
#                                                              Meta_train, train_idx, bond_type,
#                                                              fold_count, feature='sd')
#             X_train, X_valid, X_test_type = fit_meta_feature(X_train, X_valid, X_test_type,
#                                                              Meta_train, train_idx, bond_type,
#                                                              fold_count, feature='pso')
#             X_train, X_valid, X_test_type = fit_meta_feature(X_train, X_valid, X_test_type,
#                                                              Meta_train, train_idx, bond_type,
#                                                              fold_count, feature='dso')
            DEPTH = 7
            update_tracking(run_id, 'depth', DEPTH)
            train_dataset = Pool(data=X_train, label=y_train)
            valid_dataset = Pool(data=X_valid, label=y_valid)
            test_dataset = Pool(data=X_test_type)
            model = CatBoostRegressor(iterations=N_ESTIMATORS,
                                         learning_rate=LEARNING_RATE,
                                         depth=DEPTH,
                                         eval_metric=EVAL_METRIC,
                                         verbose=VERBOSE,
                                         random_state = RANDOM_STATE,
                                         thread_count=N_THREADS,
                                         # loss_function=EVAL_METRIC,
                                         # bootstrap_type='Poisson',
                                         # bagging_temperature=5,
                                         task_type = "GPU") # Train on GPU

            model.fit(train_dataset,
                      eval_set=valid_dataset,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            now = timer()
            update_tracking(run_id, '{}_tr_sec_f{}'.format(bond_type, fold_n+1),
                            (now-fold_start), integer=True)
            logger.info('Saving model file')
            model.save_model('models/{}/{}-{}-{}-{}.model'.format(MODEL_NUMBER,
                                                                  MODEL_NUMBER,
                                                               run_id,
                                                               bond_type,
                                                               fold_count))
            pred_start = timer()
            logger.info('Predicting on validation set')
            y_pred_valid = model.predict(valid_dataset)
            logger.info('Predicting on test set')
            y_pred = model.predict(test_dataset)
            now = timer()
            update_tracking(run_id, '{}_pred_sec_f{}'.format(bond_type, fold_n+1),
                            (now-pred_start), integer=True)
            update_tracking(run_id, '{}_f{}_best_iter'.format(bond_type, fold_n+1),
                            model.best_iteration_, integer=True)
            # feature importance
            logger.info('Storing the fold importance')
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = model.feature_names_
            fold_importance["importance"] = model.get_feature_importance()
            fold_importance["type"] = bond_type
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0)
            fold_score = mean_absolute_error(y_valid, y_pred_valid)
            bond_scores.append(fold_score)
            update_tracking(run_id, '{}cv_f{}'.format(bond_type, fold_n+1), fold_score, integer=False)
            logger.info('CV mean score: {0:.4f}, std: {1:.4f}.'.format(
                np.mean(bond_scores), np.std(bond_scores)))
            oof[valid_idx] = y_pred_valid.reshape(-1,)
            prediction_type += y_pred
        now = timer()
        logger.info('Completed training and predicting for bond {} fold {}-of-{} in {:0.4f} seconds'.format(bond_type,
                                                                                                            fold_count,
                                                                                                            N_FOLDS,
                                                                                                            now-fold_start))
    update_tracking(run_id, f'{bond_type}_mae_cv', np.mean(bond_scores), digits=4)
    update_tracking(run_id, f'{bond_type}_std_mae_cv', np.std(bond_scores), digits=6)
    oof_df.loc[oof_df['type'] == bond_type, 'oof_preds'] = oof
    if RUN_SINGLE_FOLD is False:
        prediction_type /= folds.n_splits
    test_pred_df.loc[test_pred_df['type'] ==
                     bond_type, 'prediction'] = prediction_type

    pred_pq_name = '{}_{}_sub_{:0.4f}_{}_{}folds_{}iter_{}lr.parquet'.format(MODEL_NUMBER,
                                                                             run_id,
                                                                             np.mean(bond_scores),
                                                                             MODEL_TYPE,
                                                                             N_FOLDS,
                                                                             N_ESTIMATORS,
                                                                             LEARNING_RATE)
    test_pred_df.to_parquet('type_results/{}/{}'.format(bond_type, pred_pq_name))
    oof_pq_name = '{}_{}_oof_{:0.4f}_{}_{}folds_{}iter_{}lr.parquet'.format(MODEL_NUMBER,
                                                                            run_id,
                                                                            np.mean(bond_scores),
                                                                            MODEL_TYPE,
                                                                            N_FOLDS,
                                                                            N_ESTIMATORS,
                                                                            LEARNING_RATE)
    fi_pq_name = '{}_{}_fi_{:0.4f}_{}_{}folds_{}iter_{}lr.parquet'.format(MODEL_NUMBER,
                                                                            run_id,
                                                                            np.mean(bond_scores),
                                                                            MODEL_TYPE,
                                                                            N_FOLDS,
                                                                            N_ESTIMATORS,
                                                                            LEARNING_RATE)

    oof_df.loc[oof_df['type'] == bond_type].to_parquet('type_results/{}/{}'.format(bond_type, oof_pq_name))
    feature_importance.loc[feature_importance['type'] == bond_type].to_parquet('type_results/{}/{}'.format(bond_type,
                                                                                                       fi_pq_name))
    now = timer()
    logger.info('Completed training and predicting for bond {} in {:0.4f} seconds'.format(bond_type,
                                                                                          now-bond_start))
    if bond_count != number_of_bonds:
        csv_save_start = timer()
        # Save the results inbetween bond type because it takes a long time
        submission_csv_name = 'temp/{}/temp{}of{}_{}_{}_submission_{}_{}folds_{}iter_{}lr.csv'.format(
            MODEL_NUMBER, bond_count, number_of_bonds, MODEL_NUMBER,
            run_id, MODEL_TYPE, N_FOLDS, N_ESTIMATORS, LEARNING_RATE)
        oof_csv_name = 'temp/{}/temp{}of{}_{}_{}_oof_{}_{}folds_{}iter_{}lr.csv'.format(
            MODEL_NUMBER, bond_count, number_of_bonds, MODEL_NUMBER,
            run_id, MODEL_TYPE, N_FOLDS, N_ESTIMATORS, LEARNING_RATE)
        fi_csv_name = 'temp/{}/temp{}of{}_{}_{}_fi_{}_{}folds_{}iter_{}lr.csv'.format(
                MODEL_NUMBER, bond_count, number_of_bonds, MODEL_NUMBER,
            run_id, MODEL_TYPE, N_FOLDS, N_ESTIMATORS, LEARNING_RATE)

        logger.info('Saving Temporary LGB Submission files:')
        logger.info(submission_csv_name)
        ss = pd.read_csv('input/sample_submission.csv')
        ss['scalar_coupling_constant'] = test_pred_df['prediction']
        ss.to_csv(submission_csv_name, index=False)
        # OOF
        oof_df.to_csv(oof_csv_name, index=False)
        # Feature Importance
        feature_importance.to_csv(fi_csv_name, index=False)
        now = timer()
        update_tracking(run_id, '{}_csv_save_sec'.format(bond_type),
                        (now-csv_save_start), integer=True)
    bond_count += 1
oof_score = mean_absolute_error(
    oof_df['scalar_coupling_constant'], oof_df['oof_preds'])
update_tracking(run_id, 'oof_score', oof_score, digits=4)
logger.info('Out of fold score is {:.4f}'.format(oof_score))

oof_gml_score = group_mean_log_mae(
    oof_df['scalar_coupling_constant'], oof_df['oof_preds'], oof_df['type'])
update_tracking(run_id, 'gml_oof_score', oof_gml_score, digits=4)
logger.info('Out of fold group mean log mae score is {:.4f}'.format(oof_gml_score))

#####################
# SAVE RESULTS
#####################
# Save Prediction and name appropriately
submission_csv_name = 'submissions/{}_{}_submission_lgb_{}folds_{:.4f}CV_{}iter_{}lr.csv'.format(MODEL_NUMBER,
                                                                                                 run_id,
                                                                                                 N_FOLDS,
                                                                                                 oof_gml_score,
                                                                                                 N_ESTIMATORS,
                                                                                                 LEARNING_RATE)
oof_csv_name = 'oof/{}_{}_oof_{}_{}folds_{:.4f}CV_{}iter_{}lr.csv'.format(
    MODEL_NUMBER, run_id, MODEL_TYPE, N_FOLDS, oof_gml_score, N_ESTIMATORS, LEARNING_RATE)
fi_csv_name = 'fi/{}_{}_fi_lgb_{}folds_{:.4f}CV_{}iter_{}lr.csv'.format(
    MODEL_NUMBER, run_id, MODEL_TYPE, N_FOLDS, oof_gml_score, N_ESTIMATORS, LEARNING_RATE)

logger.info('Saving LGB Submission as:')
logger.info(submission_csv_name)
ss = pd.read_csv('input/sample_submission.csv')
ss['scalar_coupling_constant'] = test_pred_df['prediction']
ss.to_csv(submission_csv_name, index=False)
ss.head()
# OOF
oof_df.to_csv(oof_csv_name, index=False)
# Feature Importance
feature_importance.to_csv(fi_csv_name, index=False)
end = timer()
update_tracking(run_id, 'training_time', (end-start), integer=True)
logger.info('==== Training done in {} seconds ======'.format(end - start))
logger.info('Done!')
