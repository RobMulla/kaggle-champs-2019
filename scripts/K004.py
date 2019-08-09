import logging
import sys
def get_logger():
    """
        credits to: https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    """
    FORMAT = "[%(asctime)s] %(levelname)s : %(message)s"
    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()

import pandas as pd
import os
import time
import logging
import sys
from tqdm import tqdm
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from keras.layers import Dense, Input, Activation
from keras.layers import BatchNormalization,Add,Dropout
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import callbacks
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)
import os
import gc

from datetime import datetime
from sklearn.metrics import mean_absolute_error

df_struct=pd.read_csv('../input/structures.csv')
df_train_sub_charge=pd.read_csv('../input/mulliken_charges.csv')
df_train_sub_tensor=pd.read_csv('../input/magnetic_shielding_tensors.csv')
train = pd.read_csv('../input/train.csv')

### CONFIGURABLES #######
bond_type = '3JHH'
MODEL_NUMBER = 'K004'
#########################

def plot_history(history, label):
    plt.figure(figsize=(15,5))
    plt.plot(history.history['loss'][-100:])
    plt.plot(history.history['val_loss'][-100:])
    plt.title('Loss for %s' % label)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    _= plt.legend(['Train','Validation'], loc='upper left')
    plt.show()

