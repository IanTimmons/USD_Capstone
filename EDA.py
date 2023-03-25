"""
Author: Rob Salamon
Purpose: 

Description:


Inputs:
data_dir: Folder location of dataset on local machine

"""

import mediapipe as mp

import tensorflow as tf
import tensorflow_io as tfio
import pandas as pd
import numpy as np
import sklearn

import json

import random
import io
import os

import matplotlib.pyplot as plt
import seaborn as sns

def set_random(seed=43):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_random()

cwd = os.getcwd() 
data_dir = (cwd + "\data")
print(data_dir)

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
#train_df["path"] = data_dir + "\" + train_df["path"]




