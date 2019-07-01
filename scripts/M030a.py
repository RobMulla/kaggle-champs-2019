'''
Created by: Rob Mulla
Jul 1
New Changes:
    - lgbm
    - learning rate to 0.5
    - 3 folds
Old Changes:
    - Remove features per type if feature is all nulls
    - catboost
    - learning rate to 0.1
    - PART A and PART B - one for each fold
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
# MODEL NUMBER
MODEL_NUMBER = 'M030'
script_name = os.path.basename(__file__).split('.')[0]
if MODEL_NUMBER not in script_name:
    logger.error('Model Number is not same as script! Update before running')
    raise SystemExit('Model Number is not same as script! Update before running')

# Make a runid that is unique to the time this is run for easy tracking later
run_id = "{:%m%d_%H%M}".format(datetime.now())
LEARNING_RATE = 0.05
RUN_SINGLE_FOLD = 1 # Fold number to run starting with 1 - Set to False to run all folds
FEATURES = [
            #'id',
            # 'molecule_name',
            # 'atom_index_0',
            # 'atom_index_1',
             'type',
            # 'scalar_coupling_constant',
             # 'atom0_atomic_mass',
             # 'atom0_atomic_number',
             # 'exact_mass_x',
             # 'atom0_valence',
             # 'atom0_spin_multiplicity',
             # 'atom1_atomic_mass',
             # 'atom1_atomic_number',
             # 'exact_mass_y',
             'atom1_valence',
             'atom1_spin_multiplicity',
             'left_middle_average_angle',
             'right_middle_average_angle',
             'distance',
             'is_bond',
            # 'closest_to_0',
            # '2nd_closest_to_0',
            # '3rd_closest_to_0',
            # '4th_closest_to_0',
            # '5th_closest_to_0',
            # '6th_closest_to_0',
            # '7th_closest_to_0',
            # '8th_closest_to_0',
            # '9th_closest_to_0',
            # '10th_closest_to_0',
            # 'closest_to_1',
            # '2nd_closest_to_1',
            # '3rd_closest_to_1',
            # '4th_closest_to_1',
            # '5th_closest_to_1',
            # '6th_closest_to_1',
            # '7th_closest_to_1',
            # '8th_closest_to_1',
            # '9th_closest_to_1',
            # '10th_closest_to_1',
             # 'is_closest_pair',
             'distance_closest_to_0',
             # 'is_bond_closest_to_0',
             'distance_2nd_closest_to_0',
             # 'is_bond_2nd_closest_to_0',
             'distance_3rd_closest_to_0',
             # 'is_bond_3rd_closest_to_0',
             'distance_4th_closest_to_0',
             # 'is_bond_4th_closest_to_0',
             'distance_5th_closest_to_0',
             # 'is_bond_5th_closest_to_0',
             'distance_6th_closest_to_0',
             # 'is_bond_6th_closest_to_0',
             'distance_7th_closest_to_0',
             # 'is_bond_7th_closest_to_0',
             'distance_8th_closest_to_0',
             # 'is_bond_8th_closest_to_0',
             'distance_9th_closest_to_0',
             # 'is_bond_9th_closest_to_0',
             'distance_10th_closest_to_0',
             # 'is_bond_10th_closest_to_0',
             'distance_closest_to_1',
             # 'is_bond_closest_to_1',
             'distance_2nd_closest_to_1',
             # 'is_bond_2nd_closest_to_1',
             'distance_3rd_closest_to_1',
             'is_bond_3rd_closest_to_1',
             'distance_4th_closest_to_1',
             # 'is_bond_4th_closest_to_1',
             'distance_5th_closest_to_1',
             # 'is_bond_5th_closest_to_1',
             'distance_6th_closest_to_1',
             # 'is_bond_6th_closest_to_1',
             'distance_7th_closest_to_1',
             # 'is_bond_7th_closest_to_1',
             'distance_8th_closest_to_1',
             # 'is_bond_8th_closest_to_1',
             'distance_9th_closest_to_1',
             # 'is_bond_9th_closest_to_1',
             'distance_10th_closest_to_1',
             # 'is_bond_10th_closest_to_1',
             'closest_to_0_atomic_mass',
             # 'closest_to_0_atomic_number',
             # 'closest_to_0_exact_mass',
             'closest_to_0_valence',
             'closest_to_0_spin_multiplicity',
             '2nd_closest_to_0_atomic_mass',
             # '2nd_closest_to_0_atomic_number',
             # '2nd_closest_to_0_exact_mass',
             '2nd_closest_to_0_valence',
             '2nd_closest_to_0_spin_multiplicity',
             '3rd_closest_to_0_atomic_mass',
             # '3rd_closest_to_0_atomic_number',
             # '3rd_closest_to_0_exact_mass',
             '3rd_closest_to_0_valence',
             '3rd_closest_to_0_spin_multiplicity',
             '4th_closest_to_0_atomic_mass',
             # '4th_closest_to_0_atomic_number',
             # '4th_closest_to_0_exact_mass',
             '4th_closest_to_0_valence',
             '4th_closest_to_0_spin_multiplicity',
             '5th_closest_to_0_atomic_mass',
             # '5th_closest_to_0_atomic_number',
             # '5th_closest_to_0_exact_mass',
             '5th_closest_to_0_valence',
             '5th_closest_to_0_spin_multiplicity',
             '6th_closest_to_0_atomic_mass',
             # '6th_closest_to_0_atomic_number',
             # '6th_closest_to_0_exact_mass',
             '6th_closest_to_0_valence',
             '6th_closest_to_0_spin_multiplicity',
             '7th_closest_to_0_atomic_mass',
             # '7th_closest_to_0_atomic_number',
             # '7th_closest_to_0_exact_mass',
             '7th_closest_to_0_valence',
             '7th_closest_to_0_spin_multiplicity',
             '8th_closest_to_0_atomic_mass',
             # '8th_closest_to_0_atomic_number',
             # '8th_closest_to_0_exact_mass',
             '8th_closest_to_0_valence',
             '8th_closest_to_0_spin_multiplicity',
             '9th_closest_to_0_atomic_mass',
             # '9th_closest_to_0_atomic_number',
             # '9th_closest_to_0_exact_mass',
             '9th_closest_to_0_valence',
             '9th_closest_to_0_spin_multiplicity',
             '10th_closest_to_0_atomic_mass',
             # '10th_closest_to_0_atomic_number',
             # '10th_closest_to_0_exact_mass',
             '10th_closest_to_0_valence',
             '10th_closest_to_0_spin_multiplicity',
             'closest_to_1_atomic_mass',
             # 'closest_to_1_atomic_number',
             # 'closest_to_1_exact_mass',
             'closest_to_1_valence',
             'closest_to_1_spin_multiplicity',
             '2nd_closest_to_1_atomic_mass',
             # '2nd_closest_to_1_atomic_number',
             # '2nd_closest_to_1_exact_mass',
             '2nd_closest_to_1_valence',
             '2nd_closest_to_1_spin_multiplicity',
             '3rd_closest_to_1_atomic_mass',
             # '3rd_closest_to_1_atomic_number',
             # '3rd_closest_to_1_exact_mass',
             '3rd_closest_to_1_valence',
             '3rd_closest_to_1_spin_multiplicity',
             '4th_closest_to_1_atomic_mass',
             # '4th_closest_to_1_atomic_number',
             # '4th_closest_to_1_exact_mass',
             '4th_closest_to_1_valence',
             '4th_closest_to_1_spin_multiplicity',
             '5th_closest_to_1_atomic_mass',
             # '5th_closest_to_1_atomic_number',
             # '5th_closest_to_1_exact_mass',
             '5th_closest_to_1_valence',
             '5th_closest_to_1_spin_multiplicity',
             '6th_closest_to_1_atomic_mass',
             # '6th_closest_to_1_atomic_number',
             # '6th_closest_to_1_exact_mass',
             '6th_closest_to_1_valence',
             '6th_closest_to_1_spin_multiplicity',
             '7th_closest_to_1_atomic_mass',
             # '7th_closest_to_1_atomic_number',
             # '7th_closest_to_1_exact_mass',
             '7th_closest_to_1_valence',
             '7th_closest_to_1_spin_multiplicity',
             '8th_closest_to_1_atomic_mass',
             # '8th_closest_to_1_atomic_number',
             # '8th_closest_to_1_exact_mass',
             '8th_closest_to_1_valence',
             '8th_closest_to_1_spin_multiplicity',
             '9th_closest_to_1_atomic_mass',
             # '9th_closest_to_1_atomic_number',
             # '9th_closest_to_1_exact_mass',
             '9th_closest_to_1_valence',
             '9th_closest_to_1_spin_multiplicity',
             '10th_closest_to_1_atomic_mass',
             # '10th_closest_to_1_atomic_number',
             # '10th_closest_to_1_exact_mass',
             '10th_closest_to_1_valence',
             '10th_closest_to_1_spin_multiplicity',
             'tor_ang_2leftleft_mean',
             'tor_ang_2leftleft_min',
             'tor_ang_2leftleft_max',
             'tor_ang_2leftleft_count',
             'tor_ang_2leftright_mean',
             'tor_ang_2leftright_min',
             'tor_ang_2leftright_max',
             'tor_ang_2leftright_count',
             'mol_wt',
             'num_atoms',
             'num_bonds',
             # '11th_closest_to_0',
             # '12th_closest_to_0',
             # '13th_closest_to_0',
             # '14th_closest_to_0',
             # '15th_closest_to_0',
             # '16th_closest_to_0',
             # '17th_closest_to_0',
             # '18th_closest_to_0',
             # '19th_closest_to_0',
             # '20th_closest_to_0',
             # '21st_closest_to_0',
             # '22nd_closest_to_0',
             # '23rd_closest_to_0',
             # '24th_closest_to_0',
             # '25th_closest_to_0',
             # '26th_closest_to_0',
             # '27th_closest_to_0',
             # '28th_closest_to_0',
             # '11th_closest_to_1',
             # '12th_closest_to_1',
             # '13th_closest_to_1',
             # '14th_closest_to_1',
             # '15th_closest_to_1',
             # '16th_closest_to_1',
             # '17th_closest_to_1',
             # '18th_closest_to_1',
             # '19th_closest_to_1',
             # '20th_closest_to_1',
             # '21st_closest_to_1',
             # '22nd_closest_to_1',
             # '23rd_closest_to_1',
             # '24th_closest_to_1',
             # '25th_closest_to_1',
             # '26th_closest_to_1',
             # '27th_closest_to_1',
             # '28th_closest_to_1',
             'distance_12th_closest_to_0',
             # 'is_bond_12th_closest_to_0',
             'distance_13th_closest_to_0',
             # 'is_bond_13th_closest_to_0',
             'distance_14th_closest_to_0',
             # 'is_bond_14th_closest_to_0',
             'distance_15th_closest_to_0',
             # 'is_bond_15th_closest_to_0',
             'distance_16th_closest_to_0',
             # 'is_bond_16th_closest_to_0',
             'distance_17th_closest_to_0',
             # 'is_bond_17th_closest_to_0',
             'distance_18th_closest_to_0',
             # 'is_bond_18th_closest_to_0',
             'distance_19th_closest_to_0',
             # 'is_bond_19th_closest_to_0',
             'distance_20th_closest_to_0',
             # 'is_bond_20th_closest_to_0',
             'distance_21st_closest_to_0',
             # 'is_bond_21st_closest_to_0',
             'distance_22nd_closest_to_0',
             # 'is_bond_22nd_closest_to_0',
             'distance_23rd_closest_to_0',
             # 'is_bond_23rd_closest_to_0',
             'distance_24th_closest_to_0',
             # 'is_bond_24th_closest_to_0',
             'distance_25th_closest_to_0',
             # 'is_bond_25th_closest_to_0',
             'distance_26th_closest_to_0',
             # 'is_bond_26th_closest_to_0',
             'distance_27th_closest_to_0',
             # 'is_bond_27th_closest_to_0',
             'distance_28th_closest_to_0',
             # 'is_bond_28th_closest_to_0',
             'distance_12th_closest_to_1',
             # 'is_bond_12th_closest_to_1',
             'distance_13th_closest_to_1',
             # 'is_bond_13th_closest_to_1',
             'distance_14th_closest_to_1',
             # 'is_bond_14th_closest_to_1',
             'distance_15th_closest_to_1',
             # 'is_bond_15th_closest_to_1',
             'distance_16th_closest_to_1',
             # 'is_bond_16th_closest_to_1',
             'distance_17th_closest_to_1',
             # 'is_bond_17th_closest_to_1',
             'distance_18th_closest_to_1',
             # 'is_bond_18th_closest_to_1',
             'distance_19th_closest_to_1',
             # 'is_bond_19th_closest_to_1',
             'distance_20th_closest_to_1',
             # 'is_bond_20th_closest_to_1',
             'distance_21st_closest_to_1',
             # 'is_bond_21st_closest_to_1',
             'distance_22nd_closest_to_1',
             # 'is_bond_22nd_closest_to_1',
             'distance_23rd_closest_to_1',
             # 'is_bond_23rd_closest_to_1',
             'distance_24th_closest_to_1',
             # 'is_bond_24th_closest_to_1',
             'distance_25th_closest_to_1',
             # 'is_bond_25th_closest_to_1',
             'distance_26th_closest_to_1',
             # 'is_bond_26th_closest_to_1',
             'distance_27th_closest_to_1',
             # 'is_bond_27th_closest_to_1',
             'distance_28th_closest_to_1',
             # 'is_bond_28th_closest_to_1',
             '11th_closest_to_0_atomic_mass',
             # '11th_closest_to_0_atomic_number',
             # '11th_closest_to_0_exact_mass',
             '11th_closest_to_0_valence',
             '11th_closest_to_0_spin_multiplicity',
             '12th_closest_to_0_atomic_mass',
             # '12th_closest_to_0_atomic_number',
             # '12th_closest_to_0_exact_mass',
             '12th_closest_to_0_valence',
             '12th_closest_to_0_spin_multiplicity',
             '13th_closest_to_0_atomic_mass',
             # '13th_closest_to_0_atomic_number',
             # '13th_closest_to_0_exact_mass',
             '13th_closest_to_0_valence',
             '13th_closest_to_0_spin_multiplicity',
             '14th_closest_to_0_atomic_mass',
             # '14th_closest_to_0_atomic_number',
             # '14th_closest_to_0_exact_mass',
             '14th_closest_to_0_valence',
             '14th_closest_to_0_spin_multiplicity',
             '15th_closest_to_0_atomic_mass',
             # '15th_closest_to_0_atomic_number',
             # '15th_closest_to_0_exact_mass',
             '15th_closest_to_0_valence',
             '15th_closest_to_0_spin_multiplicity',
             '16th_closest_to_0_atomic_mass',
             # '16th_closest_to_0_atomic_number',
             # '16th_closest_to_0_exact_mass',
             '16th_closest_to_0_valence',
             '16th_closest_to_0_spin_multiplicity',
             '17th_closest_to_0_atomic_mass',
             # '17th_closest_to_0_atomic_number',
             # '17th_closest_to_0_exact_mass',
             '17th_closest_to_0_valence',
             '17th_closest_to_0_spin_multiplicity',
             '18th_closest_to_0_atomic_mass',
             # '18th_closest_to_0_atomic_number',
             # '18th_closest_to_0_exact_mass',
             '18th_closest_to_0_valence',
             '18th_closest_to_0_spin_multiplicity',
             '19th_closest_to_0_atomic_mass',
             # '19th_closest_to_0_atomic_number',
             # '19th_closest_to_0_exact_mass',
             '19th_closest_to_0_valence',
             '19th_closest_to_0_spin_multiplicity',
             '20th_closest_to_0_atomic_mass',
             # '20th_closest_to_0_atomic_number',
             # '20th_closest_to_0_exact_mass',
             '20th_closest_to_0_valence',
             '20th_closest_to_0_spin_multiplicity',
             '21st_closest_to_0_atomic_mass',
             # '21st_closest_to_0_atomic_number',
             # '21st_closest_to_0_exact_mass',
             '21st_closest_to_0_valence',
             '21st_closest_to_0_spin_multiplicity',
             '22nd_closest_to_0_atomic_mass',
             # '22nd_closest_to_0_atomic_number',
             # '22nd_closest_to_0_exact_mass',
             '22nd_closest_to_0_valence',
             '22nd_closest_to_0_spin_multiplicity',
             '23rd_closest_to_0_atomic_mass',
             # '23rd_closest_to_0_atomic_number',
             # '23rd_closest_to_0_exact_mass',
             '23rd_closest_to_0_valence',
             '23rd_closest_to_0_spin_multiplicity',
             '24th_closest_to_0_atomic_mass',
             # '24th_closest_to_0_atomic_number',
             # '24th_closest_to_0_exact_mass',
             '24th_closest_to_0_valence',
             '24th_closest_to_0_spin_multiplicity',
             '25th_closest_to_0_atomic_mass',
             # '25th_closest_to_0_atomic_number',
             # '25th_closest_to_0_exact_mass',
             '25th_closest_to_0_valence',
             '25th_closest_to_0_spin_multiplicity',
             '26th_closest_to_0_atomic_mass',
             # '26th_closest_to_0_atomic_number',
             # '26th_closest_to_0_exact_mass',
             '26th_closest_to_0_valence',
             '26th_closest_to_0_spin_multiplicity',
             '27th_closest_to_0_atomic_mass',
             # '27th_closest_to_0_atomic_number',
             # '27th_closest_to_0_exact_mass',
             '27th_closest_to_0_valence',
             '27th_closest_to_0_spin_multiplicity',
             '28th_closest_to_0_atomic_mass',
             # '28th_closest_to_0_atomic_number',
             # '28th_closest_to_0_exact_mass',
             '28th_closest_to_0_valence',
             '28th_closest_to_0_spin_multiplicity',
             '11th_closest_to_1_atomic_mass',
             # '11th_closest_to_1_atomic_number',
             # '11th_closest_to_1_exact_mass',
             '11th_closest_to_1_valence',
             '11th_closest_to_1_spin_multiplicity',
             '12th_closest_to_1_atomic_mass',
             # '12th_closest_to_1_atomic_number',
             # '12th_closest_to_1_exact_mass',
             '12th_closest_to_1_valence',
             '12th_closest_to_1_spin_multiplicity',
             '13th_closest_to_1_atomic_mass',
             # '13th_closest_to_1_atomic_number',
             # '13th_closest_to_1_exact_mass',
             '13th_closest_to_1_valence',
             '13th_closest_to_1_spin_multiplicity',
             '14th_closest_to_1_atomic_mass',
             # '14th_closest_to_1_atomic_number',
             # '14th_closest_to_1_exact_mass',
             '14th_closest_to_1_valence',
             '14th_closest_to_1_spin_multiplicity',
             '15th_closest_to_1_atomic_mass',
             # '15th_closest_to_1_atomic_number',
             # '15th_closest_to_1_exact_mass',
             '15th_closest_to_1_valence',
             '15th_closest_to_1_spin_multiplicity',
             '16th_closest_to_1_atomic_mass',
             # '16th_closest_to_1_atomic_number',
             # '16th_closest_to_1_exact_mass',
             '16th_closest_to_1_valence',
             # '16th_closest_to_1_spin_multiplicity',
             '17th_closest_to_1_atomic_mass',
             # '17th_closest_to_1_atomic_number',
             # '17th_closest_to_1_exact_mass',
             '17th_closest_to_1_valence',
             '17th_closest_to_1_spin_multiplicity',
             '18th_closest_to_1_atomic_mass',
             # '18th_closest_to_1_atomic_number',
             # '18th_closest_to_1_exact_mass',
             '18th_closest_to_1_valence',
             '18th_closest_to_1_spin_multiplicity',
             '19th_closest_to_1_atomic_mass',
             # '19th_closest_to_1_atomic_number',
             # '19th_closest_to_1_exact_mass',
             '19th_closest_to_1_valence',
             '19th_closest_to_1_spin_multiplicity',
             '20th_closest_to_1_atomic_mass',
             # '20th_closest_to_1_atomic_number',
             # '20th_closest_to_1_exact_mass',
             '20th_closest_to_1_valence',
             '20th_closest_to_1_spin_multiplicity',
             '21st_closest_to_1_atomic_mass',
             # '21st_closest_to_1_atomic_number',
             # '21st_closest_to_1_exact_mass',
             '21st_closest_to_1_valence',
             '21st_closest_to_1_spin_multiplicity',
             '22nd_closest_to_1_atomic_mass',
             # '22nd_closest_to_1_atomic_number',
             # '22nd_closest_to_1_exact_mass',
             '22nd_closest_to_1_valence',
             '22nd_closest_to_1_spin_multiplicity',
             '23rd_closest_to_1_atomic_mass',
             # '23rd_closest_to_1_atomic_number',
             # '23rd_closest_to_1_exact_mass',
             '23rd_closest_to_1_valence',
             '23rd_closest_to_1_spin_multiplicity',
             '24th_closest_to_1_atomic_mass',
             # '24th_closest_to_1_atomic_number',
             # '24th_closest_to_1_exact_mass',
             '24th_closest_to_1_valence',
             '24th_closest_to_1_spin_multiplicity',
             '25th_closest_to_1_atomic_mass',
             # '25th_closest_to_1_atomic_number',
             # '25th_closest_to_1_exact_mass',
             '25th_closest_to_1_valence',
             '25th_closest_to_1_spin_multiplicity',
             '26th_closest_to_1_atomic_mass',
             # '26th_closest_to_1_atomic_number',
             # '26th_closest_to_1_exact_mass',
             '26th_closest_to_1_valence',
             '26th_closest_to_1_spin_multiplicity',
             '27th_closest_to_1_atomic_mass',
             # '27th_closest_to_1_atomic_number',
             # '27th_closest_to_1_exact_mass',
             '27th_closest_to_1_valence',
             '27th_closest_to_1_spin_multiplicity',
             '28th_closest_to_1_atomic_mass',
             # '28th_closest_to_1_atomic_number',
             # '28th_closest_to_1_exact_mass',
             '28th_closest_to_1_valence',
             '28th_closest_to_1_spin_multiplicity'
             ]

TARGET = 'scalar_coupling_constant'
N_ESTIMATORS = 5000000
VERBOSE = 1000
EARLY_STOPPING_ROUNDS = 500
RANDOM_STATE = 529
N_THREADS = 16
N_FOLDS = 3
EVAL_METRIC = 'group_mae'
#EVAL_METRIC = 'RMSE'
MODEL_TYPE = 'lgbm'
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

types = ['1JHC', '2JHH', '1JHN', '2JHN', '2JHC','3JHH','3JHC', '3JHN']
number_of_bonds = len(types)

for bond_type in types:
    # Read the files and make X, X_test, and y
    train_df = pd.read_parquet('data/FE010-train-{}.parquet'.format(bond_type)) 
    test_df = pd.read_parquet('data/FE010-test-{}.parquet'.format(bond_type)) 
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
        fold_count += 1 # First fold is 1
        if RUN_SINGLE_FOLD is not False:
            if fold_count != RUN_SINGLE_FOLD:
                logger.info('Running only for fold {}, skipping fold {}'.format(RUN_SINGLE_FOLD, fold_count))
                continue
        if MODEL_TYPE == 'lgbm':
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
            now = timer()
            update_tracking(run_id, '{}_tr_sec_f{}'.format(bond_type, fold_n+1), (now-fold_start), integer=True)
            logger.info('Saving model file')
            model.booster_.save_model('models/{}-{}-{}-{}.model'.format(MODEL_NUMBER,
                                                               run_id,
                                                               bond_type,
                                                               fold_count))
            pred_start = timer()
            logger.info('Predicting on validation set')
            y_pred_valid = model.predict(X_valid.drop('type', axis=1),
                                         num_iteration=model.best_iteration_,
                                         n_jobs=N_THREADS)
            logger.info('Predicting on test set')
            y_pred = model.predict(X_test_type.drop('type', axis=1),
                                   num_iteration=model.best_iteration_)
            now = timer()
            update_tracking(run_id, '{}_pred_sec_f{}'.format(bond_type, fold_n+1), (now-pred_start), integer=True)
            # feature importance
            logger.info('Storing the fold importance')
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X_train.drop('type', axis=1).columns
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
            y_train, y_valid = y_type.iloc[train_idx], y_type.iloc[valid_idx]
            train_dataset = Pool(data=X_train.drop('type', axis=1), label=y_train)
            valid_dataset = Pool(data=X_valid.drop('type', axis=1), label=y_valid)
            test_dataset = Pool(data=X_test_type.drop('type', axis=1))
            DEPTH = 4
            update_tracking(run_id, 'depth', DEPTH)
            model = CatBoostRegressor(iterations=N_ESTIMATORS,
                                         learning_rate=LEARNING_RATE,
                                         depth=DEPTH,
                                         eval_metric=EVAL_METRIC,
                                         verbose=VERBOSE,
                                         random_state = RANDOM_STATE,
                                         thread_count=N_THREADS,
                                         loss_function=EVAL_METRIC,
                                         task_type = "GPU") # Train on GPU

            model.fit(train_dataset,
                      eval_set=valid_dataset,
                      early_stopping_rounds=500)
            now = timer()
            update_tracking(run_id, '{}_tr_sec_f{}'.format(bond_type, fold_n+1), (now-fold_start), integer=True)
            logger.info('Saving model file')
            model.save_model('models/{}-{}-{}-{}.model'.format(MODEL_NUMBER,
                                                               run_id,
                                                               bond_type,
                                                               fold_count))
            pred_start = timer()
            logger.info('Predicting on validation set')
            y_pred_valid = model.predict(valid_dataset)
            logger.info('Predicting on test set')
            y_pred = model.predict(test_dataset)
            now = timer()
            update_tracking(run_id, '{}_pred_sec_f{}'.format(bond_type, fold_n+1), (now-pred_start), integer=True)
            bond_scores.append(mean_absolute_error(y_valid, y_pred_valid))
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
    now = timer()
    logger.info('Completed training and predicting for bond {} in {:0.4f} seconds'.format(bond_type,
                                                                                          now-bond_start))
    if bond_count != number_of_bonds:
        csv_save_start = timer()
        # Save the results inbetween bond type because it takes a long time
        submission_csv_name = 'temp/temp{}of{}_{}_{}_submission_lgb_{}folds_{}iter_{}lr.csv'.format(
            bond_count, number_of_bonds, MODEL_NUMBER, run_id, N_FOLDS, N_ESTIMATORS, LEARNING_RATE)
        oof_csv_name = 'temp/temp{}of{}_{}_{}_oof_lgb_{}folds_{}iter_{}lr.csv'.format(
            bond_count, number_of_bonds, MODEL_NUMBER, run_id, N_FOLDS, N_ESTIMATORS, LEARNING_RATE)
        if MODEL_TYPE == 'lgbm':
            fi_csv_name = 'temp/temp{}of{}_{}_{}_fi_lgb_{}folds_{}iter_{}lr.csv'.format(
                bond_count, number_of_bonds, MODEL_NUMBER, run_id, N_FOLDS, N_ESTIMATORS, LEARNING_RATE)

        logger.info('Saving Temporary LGB Submission files:')
        logger.info(submission_csv_name)
        ss = pd.read_csv('input/sample_submission.csv')
        ss['scalar_coupling_constant'] = test_pred_df['prediction']
        ss.to_csv(submission_csv_name, index=False)
        ss.head()
        # OOF
        oof_df.to_csv(oof_csv_name, index=False)
        # Feature Importance
        if MODEL_TYPE == 'lgbm':
            feature_importance.to_csv(fi_csv_name, index=False)
        now = timer()
        update_tracking(run_id, '{}_csv_save_sec'.format(bond_type), (now-csv_save_start), integer=True)
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
oof_csv_name = 'oof/{}_{}_oof_lgb_{}folds_{:.4f}CV_{}iter_{}lr.csv'.format(
    MODEL_NUMBER, run_id, N_FOLDS, oof_gml_score, N_ESTIMATORS, LEARNING_RATE)
fi_csv_name = 'fi/{}_{}_fi_lgb_{}folds_{:.4f}CV_{}iter_{}lr.csv'.format(
    MODEL_NUMBER, run_id, N_FOLDS, oof_gml_score, N_ESTIMATORS, LEARNING_RATE)

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
