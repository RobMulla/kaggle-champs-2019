import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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

print('Reading input files....')
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
structures = pd.read_csv('input/structures.csv')
#####################
## FEATURE CREATION
#####################
print('Creating features....')
def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train_df = map_atom_info(train_df, 0)
train_df = map_atom_info(train_df, 1)

test_df = map_atom_info(test_df, 0)
test_df = map_atom_info(test_df, 1)

# https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
train_p_0 = train_df[['x_0', 'y_0', 'z_0']].values
train_p_1 = train_df[['x_1', 'y_1', 'z_1']].values
test_p_0 = test_df[['x_0', 'y_0', 'z_0']].values
test_p_1 = test_df[['x_1', 'y_1', 'z_1']].values

train_df['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test_df['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

# make categorical variables
atom_map = {'H': 0,
            'C': 1,
            'N': 2}
train_df['atom_0_cat'] = train_df['atom_0'].map(atom_map).astype('int')
train_df['atom_1_cat'] = train_df['atom_1'].map(atom_map).astype('int')
test_df['atom_0_cat'] = test_df['atom_0'].map(atom_map).astype('int')
test_df['atom_1_cat'] = test_df['atom_1'].map(atom_map).astype('int')

# One Hot Encode the Type
train_df = pd.concat([train_df, pd.get_dummies(train_df['type'])], axis=1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['type'])], axis=1)

train_df['dist_to_type_mean'] = train_df['dist'] / train_df.groupby('type')['dist'].transform('mean')
test_df['dist_to_type_mean'] = test_df['dist'] / test_df.groupby('type')['dist'].transform('mean')

# Atom Count
atom_count_dict = structures.groupby('molecule_name').count()['atom_index'].to_dict()
train_df['atom_count'] = train_df['molecule_name'].map(atom_count_dict)
test_df['atom_count'] = test_df['molecule_name'].map(atom_count_dict)

#####################
## CONFIGURABLES
#####################


FEATURES = ['atom_index_0', 'atom_index_1',
            'atom_0_cat',
            'x_0', 'y_0', 'z_0',
            'atom_1_cat',
            'x_1', 'y_1', 'z_1', 'dist', 'dist_to_type_mean',
            'atom_count',
            # '1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN'
           ]

TARGET = 'scalar_coupling_constant'
# CAT_FEATS = ['atom_0_cat', 'atom_1_cat','type']

N_ESTIMATORS = 500000
VERBOSE = 500
EARLY_STOPPING_ROUNDS = 200
RANDOM_STATE = 529

#####################
## CREATE FINAL DATASETS
#####################

X = train_df[FEATURES]
X_test = test_df[FEATURES]
y = train_df[TARGET]

print(X.dtypes)

#####################
## TRAIN MODEL
#####################
print('Training model....')
from catboost import Pool, cv
from catboost import CatBoostRegressor, Pool

ITERATIONS = 50000

train_dataset = Pool(data=train_df[FEATURES],
                  label=train_df['scalar_coupling_constant'].values,
                  #cat_features=CAT_FEATS
                  )

cb_model = CatBoostRegressor(iterations=ITERATIONS,
                             learning_rate=0.05,
                             depth=7,
                             eval_metric='MAE',
                             random_seed = 529,
                             # task_type="GPU"
                             thread_count=32
                             )

# Fit the model
cb_model.fit(train_dataset, verbose=1000)

# Predict
test_data = test_df[FEATURES]

test_dataset = Pool(data=test_data,
                    #cat_features=CAT_FEATS
                    )

prediction = cb_model.predict(test_dataset)

#####################
# SAVE RESULTS
#####################
# Make a runid that is unique to the time this is run for easy tracking later
run_id = "{:%m%d_%H%M}".format(datetime.now())

# Save Prediction and name appropriately
n_fold = 0
submission_csv_name = 'submissions/{}_submission_cat_{}folds.csv'.format(run_id, n_fold)
# oof_csv_name = 'oof/{}oof_cat_{}folds_{:.4f}CV.csv'.format(run_id, n_fold, np.mean(scores))
# fi_csv_name = 'fi/{}fi_cat_{}folds_{:.4f}CV.csv'.format(run_id, n_fold, np.mean(scores))

print('Saving Catboost Submission as:')
print(submission_csv_name)
ss = pd.read_csv('input/sample_submission.csv')
ss['scalar_coupling_constant'] = prediction
ss.to_csv(submission_csv_name, index=False)
ss.head()
# OOF
# oof_df = train_df[['id','molecule_name','scalar_coupling_constant']].copy()
# oof_df['oof_pred'] = oof
# oof_df.to_csv(oof_csv_name, index=False)
# Feature Importance
# feature_importance.to_csv(fi_csv_name, index=False)

