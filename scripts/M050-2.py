'''
Created by: Rob Mulla
Jul 18

New Changes:
    - XGBoost
    - Higher Features Importance Threshold
Todo later:
    - Email updates.
    - Sync logs online?
    - Lower learning rate
Old Changes:
    - Features from FE019
    - QM9 features
    - Save feature importance for FC
    - Use new formatting to save temp files.
    - Best features per type
    - Increased meta-feature folds
    - Import best features function
    - 3 folds
    - Calculate meta feature for FC
    - Calculated within fold
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
import xgboost
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


# #### DELAY 2.5 Hours
# DELAY_HOURS = 0.001
# logger.info('Delaying for {:0.4f} Hours'.format(DELAY_HOURS))
# time.sleep(60 * 60 * DELAY_HOURS)
# logger.info('Done waiting! Starting program.')

####################
# CONFIGURABLES
#####################

# MODEL NUMBER
MODEL_NUMBER = "M050"
# script_name = os.path.basename(__file__).split('.')[0]
# if script_name not in MODEL_NUMBER:
#     logger.error('Model Number is not same as script! Update before running')
#     raise SystemExit('Model Number is not same as script! Update before running')

# Make a runid that is unique to the time this is run for easy tracking later
run_id = "{:%m%d_%H%M}".format(datetime.now())
LEARNING_RATE = 0.005
RUN_SINGLE_FOLD = (
    False
)  # Fold number to run starting with 1 - Set to False to run all folds
TARGET = "scalar_coupling_constant"
N_ESTIMATORS = 500000  # 500000
N_META_ESTIMATORS = 300000  # 300000
VERBOSE = 1000
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 529
N_THREADS = 48
DEPTH = 9
META_DEPTH = 9
N_FOLDS = 3
N_META_FOLDS = 2
# EVAL_METRIC = 'group_mae'
EVAL_METRIC = "mae"
MODEL_TYPE = "xgboost"
META_DEPTH = 7

xgb_params = {'colsample_bytree': 1,
         'gamma': 0,
         'learning_rate': LEARNING_RATE,
         'max_depth' : DEPTH,
         'min_child_weight': 1,
         'n_estimators': N_ESTIMATORS,
         'reg_alpha': 0,
         'reg_lambda': 1,
         'subsample': 1,
         'seed': RANDOM_STATE,
         'n_jobs': N_THREADS,
         'tree_method': "gpu_hist",
         "gpu_id": 1
        }

lgb_params = {
    "boosting_type": "gbdt",
    "objective": "regression_l2",
    "learning_rate": LEARNING_RATE,
    "num_leaves": 255,
    "sub_feature": 0.50,
    "sub_row": 0.75,
    "bagging_freq": 1,
    "metric": EVAL_METRIC,
    "random_state": RANDOM_STATE,
}

# Order shortest to longest
types = ['3JHN', '2JHH', '1JHN', '3JHH', '1JHC', '2JHN', '2JHC', '3JHC']
# types = ['3JHH', '1JHC', '2JHN', '2JHC', '3JHC']
# types = ["3JHH", "1JHC"] #, "2JHC","3JHC"]
#types = ['2JHN'] #, '2JHH', '1JHN', '3JHH', '1JHC', '2JHN', '2JHC', '3JHC']
# Reverse order for 50-2
types = ['3JHC', '2JHC', '2JHN']
#####################
## SETUP LOGGER
#####################
def get_logger():
    """
        credits to: https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    """
    os.environ["TZ"] = "US/Eastern"
    time.tzset()
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("main")
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
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def group_mean_log_mae(y_true, y_pred, groups, floor=1e-9):
    maes = (y_true - y_pred).abs().groupby(groups).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


#####################
# READ INPUT FILES
#####################
logger.info("Reading input files....")
path = "input/"
# train_df = pd.read_parquet(f'{path}FE009_train_pandas.parquet')
# test_df = pd.read_parquet(f'{path}FE009_test_pandas.parquet')
ss = pd.read_csv("input/sample_submission.csv")


##########################
# Tracking Sheet function
#########################
def update_tracking(
    run_id, field, value, csv_file="tracking/tracking.csv", integer=False, digits=None
):
    df = pd.read_csv(csv_file, index_col=[0])
    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)
    df.loc[run_id, field] = value  # Model number is index
    df.to_csv(csv_file)

update_tracking(run_id, "model_number", MODEL_NUMBER)
update_tracking(run_id, "n_estimators", N_ESTIMATORS)
update_tracking(run_id, "early_stopping_rounds", EARLY_STOPPING_ROUNDS)
update_tracking(run_id, "random_state", RANDOM_STATE)
update_tracking(run_id, "n_threads", N_THREADS)
update_tracking(run_id, "learning_rate", LEARNING_RATE)
update_tracking(run_id, "n_fold", N_FOLDS)
update_tracking(run_id, "model_type", MODEL_TYPE)
update_tracking(run_id, "eval_metric", EVAL_METRIC)

def get_good_features(bond_type):
    """
    Read csv with stored best features
    """
    good_feats = pd.read_csv("fi/FI_ANALYSIS_M049_GOODFEATS.csv", index_col=0)
    good_feats = good_feats.fillna(False)
    return good_feats.loc[good_feats[bond_type]].index.tolist()

#####################################
# FUNCTION FOR META FEATURE CREATION
#####################################
def fit_meta_feature(
    X_train,
    X_valid,
    X_test,
    Meta_train,
    train_idx,
    bond_type,
    base_fold,
    feature="fc",
    N_META_FOLDS=N_META_FOLDS,
    N_META_ESTIMATORS=N_META_ESTIMATORS,
    model_type="catboost",
):
    """
    Adds meta features to train, test and val
    """
    logger.info(f"Creating meta feature {feature}")
    logger.info(
        "X_train, X_valid and X_test are shapes {} {} {}".format(
            X_train.shape, X_valid.shape, X_test.shape
        )
    )
    folds = GroupKFold(n_splits=N_META_FOLDS)
    fold_count = 1

    # Init predictions
    X_valid["meta_" + feature] = 0
    X_test["meta_" + feature] = 0
    X_train["meta_" + feature] = 0
    X_train_oof = X_train[["meta_" + feature]].copy()
    X_train = X_train.drop("meta_" + feature, axis=1)
    feature_importance = pd.DataFrame()
    for fold_n, (train_idx2, valid_idx2) in enumerate(
        folds.split(X_train, groups=mol_group_type.iloc[train_idx].values)
    ):
        logger.info(
            "Running Meta Feature Type {} - Fold {} of {}".format(
                feature, fold_count, folds.n_splits
            )
        )
        update_tracking(
            run_id, "{}_meta_{}_est".format(bond_type, feature), N_META_ESTIMATORS
        )
        update_tracking(
            run_id, "{}_meta_{}_metafolds".format(bond_type, feature), N_META_FOLDS
        )

        X_train2 = X_train.loc[X_train.reset_index().index.isin(train_idx2)]
        X_valid2 = X_train.loc[X_train.reset_index().index.isin(valid_idx2)]
        X_train2 = X_train2.copy()
        X_valid2 = X_valid2.copy()
        y_train2 = Meta_train.loc[Meta_train.reset_index().index.isin(train_idx2)][
            feature
        ]
        y_valid2 = Meta_train.loc[Meta_train.reset_index().index.isin(valid_idx2)][
            feature
        ]
        fold_count += 1

        if model_type == "catboost":
            train_dataset = Pool(data=X_train2, label=y_train2)
            metavalid_dataset = Pool(data=X_valid2, label=y_valid2)
            valid_dataset = Pool(data=X_valid)
            test_dataset = Pool(data=X_test)
            model = CatBoostRegressor(
                iterations=N_META_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                depth=META_DEPTH,
                eval_metric=EVAL_METRIC,
                verbose=VERBOSE,
                random_state=RANDOM_STATE,
                thread_count=N_THREADS,
                task_type="GPU",
            )  # Train on GPU

            model.fit(
                train_dataset,
                eval_set=metavalid_dataset,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
            y_pred_meta_valid = model.predict(metavalid_dataset)
            y_pred_valid = model.predict(valid_dataset)
            y_pred = model.predict(test_dataset)

            X_train_oof.loc[
                X_train_oof.reset_index().index.isin(valid_idx2), "meta_" + feature
            ] = y_pred_meta_valid
            X_valid["meta_" + feature] += y_pred_valid
            X_test["meta_" + feature] += y_pred

            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X_train.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["type"] = bond_type
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0
            )
        elif model_type == "xgboost":
            model = xgboost.XGBRegressor(**xgb_params)
            model.fit(
                X_train2,
                y_train2,
                eval_metric=EVAL_METRIC,
                eval_set=[(X_valid2, y_valid2)],
                verbose=VERBOSE,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )

            y_pred_meta_valid = model.predict(X_valid2)
            y_pred_valid = model.predict(X_valid.drop("meta_" + feature, axis=1))
            y_pred = model.predict(X_test.drop("meta_" + feature, axis=1))

            X_train_oof.loc[
                X_train_oof.reset_index().index.isin(valid_idx2), "meta_" + feature
            ] = y_pred_meta_valid
            X_valid["meta_" + feature] += y_pred_valid
            X_test["meta_" + feature] += y_pred

            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X_train.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["type"] = bond_type
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0
            )
    oof_score = mean_absolute_error(Meta_train[feature], X_train_oof["meta_" + feature])
    log_oof_score = np.log(oof_score)
    logger.info(
        f"Meta feature {feature} has MAE {oof_score:0.4f} LMAE {log_oof_score:0.4f}"
    )
    update_tracking(
        run_id, "{}_meta_{}_mae_cv_f{}".format(bond_type, feature, base_fold), oof_score
    )
    update_tracking(
        run_id,
        "{}_meta_{}_lmae_cv_f{}".format(bond_type, feature, base_fold),
        log_oof_score,
    )

    feature_importance.to_parquet(
        "type_results/{}/meta/{}_{}_{}_fi_meta_{}_f{}_{:0.4f}MAE_{:0.4f}LMAE.parquet".format(
            bond_type,
            MODEL_NUMBER,
            run_id,
            bond_type,
            feature,
            base_fold,
            oof_score,
            log_oof_score,
        )
    )

    X_train_oof.to_parquet(
        "type_results/{}/meta/{}_{}_{}_oof_meta_{}_f{}_{:0.4f}MAE_{:0.4f}LMAE.parquet".format(
            bond_type,
            MODEL_NUMBER,
            run_id,
            bond_type,
            feature,
            base_fold,
            oof_score,
            log_oof_score,
        )
    )

    X_train.to_parquet(
        "type_results/{}/meta/{}_{}_{}_X_train_meta_{}_f{}_{:0.4f}MAE_{:0.4f}LMAE.parquet".format(
            bond_type,
            MODEL_NUMBER,
            run_id,
            bond_type,
            feature,
            base_fold,
            oof_score,
            log_oof_score,
        )
    )

    X_valid.to_parquet(
        "type_results/{}/meta/{}_{}_{}_X_valid_meta_{}_f{}_{:0.4f}MAE_{:0.4f}LMAE.parquet".format(
            bond_type,
            MODEL_NUMBER,
            run_id,
            bond_type,
            feature,
            base_fold,
            oof_score,
            log_oof_score,
        )
    )

    X_valid.to_parquet(
        "type_results/{}/meta/{}_{}_{}_X_valid_meta_{}_f{}_{:0.4f}MAE_{:0.4f}LMAE.parquet".format(
            bond_type,
            MODEL_NUMBER,
            run_id,
            bond_type,
            feature,
            base_fold,
            oof_score,
            log_oof_score,
        )
    )

    X_valid["meta_" + feature] = X_valid["meta_" + feature] / N_META_FOLDS
    X_test["meta_" + feature] = X_test["meta_" + feature] / N_META_FOLDS
    X_train["meta_" + feature] = X_train_oof["meta_" + feature]
    logger.info("Done creating meta features")
    logger.info(
        "X_train, X_valid and X_test are shapes {} {} {}".format(
            X_train.shape, X_valid.shape, X_test.shape
        )
    )
    return X_train, X_valid, X_test

#########################################
## FUNCTION FOR SAVING TYPE LEVEL RESULTS
#########################################
test = pd.read_csv("input/test.csv")


def save_type_data(
    type_,
    oof,
    sub,
    fi,
    MODEL_NUMBER,
    run_id,
    MODEL_TYPE,
    N_FOLDS,
    N_ESTIMATORS,
    LEARNING_RATE,
):
    """
    Saves the oof, sub, and fi files int he type_results folder with correct naming convention
    """
    oof_type = oof.loc[oof["type"] == type_]
    score = mean_absolute_error(
        oof_type["scalar_coupling_constant"], oof_type["oof_preds"]
    )
    logscore = np.log(score)
    if score > 1:
        print(f"No predictions for {type_}")
        return
    print(
        f"===== Saving results for for type {type_} - mae {score} - log mae {logscore}"
    )

    oof_type = oof.loc[oof["type"] == type_]

    sub_type = test[["id", "molecule_name", "type"]].merge(sub, on="id")
    sub_type = sub_type.loc[sub_type["type"] == type_]
    if np.sum(sub_type["scalar_coupling_constant"] == 0) > 10:
        print("ERROR! Sub has to many zero predictions")
        return
    expected_len = len(test.loc[test["type"] == type_])
    if expected_len != len(sub_type):
        print("ERRROR LENGTHS NOT THE SAME")
        return

    # Name Files and save
    fn_template = "type_results/{}/{}_{}_{}_XXXXXXX_{:0.4f}MAE_{:0.4}LMAE_{}_{}folds_{}iter_{}lr.parquet".format(
        type_,
        MODEL_NUMBER,
        run_id,
        type_,
        score,
        logscore,
        MODEL_TYPE,
        N_FOLDS,
        N_ESTIMATORS,
        LEARNING_RATE,
    )
    sub_name = fn_template.replace("XXXXXXX", "sub")
    oof_name = fn_template.replace("XXXXXXX", "oof")
    sub_type.to_parquet(sub_name)
    oof_type.to_parquet(oof_name)

    if fi is not None:
        fi_type = fi.loc[fi["type"] == type_]
        fi_name = fn_template.replace("XXXXXXX", "fi")
        print(fi_name)
        fi_type.to_parquet(fi_name)


###############
# PREPARE MODEL DATA
################
folds = GroupKFold(n_splits=N_FOLDS)

# Setup arrays for storing results
train_df = pd.read_csv("input/train.csv")
oof_df = train_df[["id", "type", "scalar_coupling_constant"]].copy()
mol_group = train_df[["molecule_name", "type"]].copy()
del train_df
gc.collect()

oof_df["oof_preds"] = 0
test_df = pd.read_csv(
    "input/test.csv"
)  # only loading for skeleton not features
prediction = np.zeros(len(test_df))
feature_importance = pd.DataFrame()
test_pred_df = test_df[["id", "type", "molecule_name"]].copy()
del test_df
gc.collect()
test_pred_df["prediction"] = 0
bond_count = 1

number_of_bonds = len(types)

#### CREATE FOLDERS FOR MODEL NUMBER IF THEY DONT EXIST

# if not os.path.exists('models/{}'.format(MODEL_NUMBER)):
#     os.makedirs('models/{}'.format(MODEL_NUMBER))
# if not os.path.exists('temp/{}'.format(MODEL_NUMBER)):
#     os.makedirs('temp/{}'.format(MODEL_NUMBER))

### Load Scalar Coupling Components
# tr_scc = pd.read_parquet('data/tr_scc.parquet')
tr_scc = pd.read_csv(
    "input/scalar_coupling_contributions.csv")

#####################
# TRAIN MODEL
#####################
logger.info("Training model....")

for bond_type in types:
    # Read the files and make X, X_test, and y
    train_df = pd.read_parquet(
        "data/FE019/FE019-train-{}.parquet".format(bond_type)
    )
    test_df = pd.read_parquet(
        "data/FE019/FE019-test-{}.parquet".format(bond_type)
    )
    if MODEL_TYPE == "xgboost":
        train_df.columns = [x.replace('[','_').replace(']','_').replace(', ','_').replace(' ','_').replace('.','') for x in train_df.columns]
        test_df.columns = [x.replace('[','_').replace(']','_').replace(', ','_').replace(' ','_').replace('.','') for x in test_df.columns]

    Meta = train_df[["molecule_name", "atom_index_0", "atom_index_1"]].merge(
        tr_scc, on=["molecule_name", "atom_index_0", "atom_index_1"]
    )[["fc", "sd", "pso", "dso"]]

    FEATURES = get_good_features(bond_type)
    FEATURES = [f for f in FEATURES if 'meta' not in f] # Drop meta feature
    # FEATURES = GOOD_FEATURES
    update_tracking(run_id, "{}_features".format(bond_type), len(FEATURES))
    logger.info('{}: Using features {}'.format(bond_type, [x for x in FEATURES]))
    X_type = train_df[FEATURES].copy()
    X_test_type = test_df[FEATURES].copy()
    y_type = train_df[TARGET].copy()
    del train_df
    del test_df
    gc.collect()
    # Remove colmns that have all nulls
    logger.info("{}: {} Features before dropping null columns".format(bond_type, len(X_type.columns)))
    X_type = X_type.dropna(how="all", axis=1)
    X_test_type = X_test_type[X_type.columns]
    logger.info("{}: {} Features after dropping null columns".format(bond_type, len(X_type.columns)))
    # Start training for type
    bond_start = timer()
    fold_count = 0  # Will be incremented at the start of the fold
    # Train the model
    # X_type = X.loc[X['type'] == bond_type]
    # y_type = y.iloc[X_type.index]
    # X_test_type = X_test.loc[X_test['type'] == bond_type]
    mol_group_type = mol_group.loc[mol_group["type"] == bond_type]["molecule_name"]
    oof = np.zeros(len(X_type))
    prediction_type = np.zeros(len(X_test_type))
    bond_scores = []
    for fold_n, (train_idx, valid_idx) in enumerate(
        folds.split(X_type, groups=mol_group_type)
    ):
        Meta_train, Meta_valid = Meta.iloc[train_idx], Meta.iloc[valid_idx]
        fold_count += 1  # First fold is 1
        if RUN_SINGLE_FOLD is not False:
            if fold_count != RUN_SINGLE_FOLD:
                logger.info(
                    "{}: Running only for fold {}, skipping fold {}".format(
                        bond_type, RUN_SINGLE_FOLD, fold_count
                    )
                )
                continue
        if MODEL_TYPE == "lgbm":
            fold_start = timer()
            logger.info(
                "{}: Running Type {} - Fold {} of {}".format(
                    bond_type, bond_type, fold_count, folds.n_splits
                )
            )
            X_train, X_valid = X_type.iloc[train_idx], X_type.iloc[valid_idx]
            y_train, y_valid = y_type.iloc[train_idx], y_type.iloc[valid_idx]
            X_train, X_valid, X_test_type = fit_meta_feature(
                X_train,
                X_valid,
                X_test_type,
                Meta_train,
                train_idx,
                bond_type,
                fold_count,
                model_type=MODEL_TYPE,
            )
            model = lgb.LGBMRegressor(
                **lgb_params, n_estimators=N_ESTIMATORS, n_jobs=N_THREADS
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],  # (X_train, y_train),
                eval_metric=EVAL_METRIC,
                verbose=VERBOSE,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
            now = timer()
            update_tracking(
                run_id,
                "{}_tr_sec_f{}".format(bond_type, fold_n + 1),
                (now - fold_start),
                integer=True,
            )
            logger.info("{}: Saving model file".format(bond_type))
            model.booster_.save_model(
                "models/{}-{}-{}-{}.model".format(
                    MODEL_NUMBER, run_id, bond_type, fold_count
                )
            )
            pred_start = timer()
            logger.info("{}: Predicting on validation set".format(bond_type))
            y_pred_valid = model.predict(
                X_valid, num_iteration=model.best_iteration_, n_jobs=N_THREADS
            )
            logger.info("{}: Predicting on test set".format(bond_type))
            y_pred = model.predict(X_test_type, num_iteration=model.best_iteration_)
            now = timer()
            update_tracking(
                run_id,
                "{}_pred_sec_f{}".format(bond_type, fold_n + 1),
                (now - pred_start),
                integer=True,
            )
            update_tracking(
                run_id,
                "{}_f{}_best_iter".format(bond_type, fold_n + 1),
                model.best_iteration_,
                integer=True,
            )

            # feature importance
            logger.info("{}: Storing the fold importance".format(bond_type))
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X_train.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["type"] = bond_type
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0
            )

            bond_scores.append(mean_absolute_error(y_valid, y_pred_valid))
            logger.info(
                "{}: CV mean score: {:.4f}, std: {:.4f}.".format(
                    bond_type, np.mean(bond_scores), np.std(bond_scores)
                )
            )
            oof[valid_idx] = y_pred_valid.reshape(-1)
            prediction_type += y_pred
        elif MODEL_TYPE == "catboost":
            fold_start = timer()
            logger.info(
                "{}: Running Type {} - Fold {} of {}".format(
                    bond_type, bond_type, fold_count, folds.n_splits
                )
            )
            X_train, X_valid = X_type.iloc[train_idx], X_type.iloc[valid_idx]
            X_train = X_train.copy()
            X_valid = X_valid.copy()
            y_train, y_valid = y_type.iloc[train_idx], y_type.iloc[valid_idx]

            ### ADD META FEATURES
            X_train, X_valid, X_test_type = fit_meta_feature(
                X_train,
                X_valid,
                X_test_type,
                Meta_train,
                train_idx,
                bond_type,
                fold_count,
                feature="fc",
                model_type=MODEL_TYPE,
            )
            DEPTH = 7
            update_tracking(run_id, "depth", DEPTH)
            train_dataset = Pool(data=X_train, label=y_train)
            valid_dataset = Pool(data=X_valid, label=y_valid)
            test_dataset = Pool(data=X_test_type)
            model = CatBoostRegressor(
                iterations=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                depth=DEPTH,
                eval_metric=EVAL_METRIC,
                verbose=VERBOSE,
                random_state=RANDOM_STATE,
                thread_count=N_THREADS,
                # loss_function=EVAL_METRIC,
                # bootstrap_type='Poisson',
                # bagging_temperature=5,
                task_type="GPU",
            )  # Train on GPU

            model.fit(
                train_dataset,
                eval_set=valid_dataset,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
            now = timer()
            update_tracking(
                run_id,
                "{}_tr_sec_f{}".format(bond_type, fold_n + 1),
                (now - fold_start),
                integer=True,
            )
            logger.info("{}: Saving model file".format(bond_type))
            model.save_model(
                "models/{}/{}-{}-{}-{}.model".format(
                    MODEL_NUMBER, MODEL_NUMBER, run_id, bond_type, fold_count
                )
            )
        elif MODEL_TYPE == "xgboost":
            fold_start = timer()
            logger.info(
                "{}: Running Type {} - Fold {} of {}".format(
                    bond_type, bond_type, fold_count, folds.n_splits
                )
            )
            X_train, X_valid = X_type.iloc[train_idx], X_type.iloc[valid_idx]
            X_train = X_train.copy()
            X_valid = X_valid.copy()
            y_train, y_valid = y_type.iloc[train_idx], y_type.iloc[valid_idx]

            ### ADD META FEATURES
            X_train, X_valid, X_test_type = fit_meta_feature(
                X_train,
                X_valid,
                X_test_type,
                Meta_train,
                train_idx,
                bond_type,
                fold_count,
                feature="fc",
                model_type="xgboost",
            )
            update_tracking(run_id, "depth", DEPTH)
            model = xgboost.XGBRegressor(**xgb_params)
            model.fit(
                X_train,
                y_train,
                eval_metric=EVAL_METRIC,
                eval_set=[(X_valid, y_valid)],
                verbose=VERBOSE,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
            now = timer()
            update_tracking(
                run_id,
                "{}_tr_sec_f{}".format(bond_type, fold_n + 1),
                (now - fold_start),
                integer=True,
            )
            logger.info("{}: Saving model file".format(bond_type))
            model.save_model(
                "models/{}/{}-{}-{}-{}.model".format(
                    MODEL_NUMBER, MODEL_NUMBER, run_id, bond_type, fold_count
                )
            )
            pred_start = timer()
            logger.info("{}: Predicting on validation set".format(bond_type))
            y_pred_valid = model.predict(X_valid)
            logger.info("{}: Predicting on test set".format(bond_type))
            y_pred = model.predict(X_test_type)
            now = timer()
            update_tracking(
                run_id,
                "{}_pred_sec_f{}".format(bond_type, fold_n + 1),
                (now - pred_start),
                integer=True,
            )
            update_tracking(
                run_id,
                "{}_f{}_best_iter".format(bond_type, fold_n + 1),
                model.get_booster().best_iteration,
                integer=True,
            )
            # feature importance
            logger.info("{}: Storing the fold importance".format(bond_type))
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X_train.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["type"] = bond_type
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0
            )
            fold_score = mean_absolute_error(y_valid, y_pred_valid)
            bond_scores.append(fold_score)
            update_tracking(
                run_id,
                "{}cv_f{}".format(bond_type, fold_n + 1),
                fold_score,
                integer=False,
            )
            logger.info(
                "{}: CV mean score: {:.4f}, std: {:.4f}.".format(
                    bond_type, np.mean(bond_scores), np.std(bond_scores)
                )
            )
            oof[valid_idx] = y_pred_valid.reshape(-1)
            prediction_type += y_pred
        now = timer()
        logger.info(
            "{}: Completed training and predicting for bond {} fold {}-of-{} in {:0.4f} seconds".format(
                bond_type, bond_type, fold_count, N_FOLDS, now - fold_start
            )
        )
    update_tracking(run_id, f"{bond_type}_mae_cv", np.mean(bond_scores), digits=4)
    update_tracking(run_id, f"{bond_type}_std_mae_cv", np.std(bond_scores), digits=6)
    oof_df.loc[oof_df["type"] == bond_type, "oof_preds"] = oof
    if RUN_SINGLE_FOLD is False:
        prediction_type /= folds.n_splits
    test_pred_df.loc[test_pred_df["type"] == bond_type, "prediction"] = prediction_type
    logger.info(
        "Completed training and predicting for bond {} in {:0.4f} seconds".format(
            bond_type, now - bond_start
        )
    )

    ## SAVE OOF, PREDS AND FI FOR TYPE
    sub = pd.read_csv("input/sample_submission.csv")
    sub["scalar_coupling_constant"] = test_pred_df["prediction"]
    save_type_data(
        bond_type,
        oof_df,
        sub,
        feature_importance,
        MODEL_NUMBER,
        run_id,
        MODEL_TYPE,
        N_FOLDS,
        N_ESTIMATORS,
        LEARNING_RATE,
    )
    bond_count += 1
oof_score = mean_absolute_error(oof_df["scalar_coupling_constant"], oof_df["oof_preds"])
update_tracking(run_id, "oof_score", oof_score, digits=4)
logger.info("Out of fold score is {:.4f}".format(oof_score))

oof_gml_score = group_mean_log_mae(
    oof_df["scalar_coupling_constant"], oof_df["oof_preds"], oof_df["type"]
)
update_tracking(run_id, "gml_oof_score", oof_gml_score, digits=4)
logger.info("Out of fold group mean log mae score is {:.4f}".format(oof_gml_score))

#####################
# SAVE RESULTS
#####################
# Save Prediction and name appropriately
# submission_csv_name = "submissions/{}_{}_submission_{}folds_{:.4f}CV_{}iter_{}lr.csv".format(
#     MODEL_NUMBER, run_id, N_FOLDS, oof_gml_score, N_ESTIMATORS, LEARNING_RATE
# )
# oof_csv_name = "oof/{}_{}_oof_{}_{}folds_{:.4f}CV_{}iter_{}lr.csv".format(
#     MODEL_NUMBER,
#     run_id,
#     MODEL_TYPE,
#     N_FOLDS,
#     oof_gml_score,
#     N_ESTIMATORS,
#     LEARNING_RATE,
# )
# fi_csv_name = "fi/{}_{}_fi_{}folds_{:.4f}CV_{}iter_{}lr.csv".format(
#     MODEL_NUMBER,
#     run_id,
#     MODEL_TYPE,
#     N_FOLDS,
#     oof_gml_score,
#     N_ESTIMATORS,
#     LEARNING_RATE,
# )

# logger.info("Saving LGB Submission as:")
# logger.info(submission_csv_name)
# ss = pd.read_csv("input/sample_submission.csv")
# ss["scalar_coupling_constant"] = test_pred_df["prediction"]
# ss.to_csv(submission_csv_name, index=False)
# ss.head()
# # OOF
# oof_df.to_csv(oof_csv_name, index=False)
# # Feature Importance
# feature_importance.to_csv(fi_csv_name, index=False)
end = timer()
update_tracking(run_id, "training_time", (end - start), integer=True)
logger.info("==== Training done in {} seconds ======".format(end - start))
logger.info("Done!")
