import numpy as np
import tensorflow as tf
import csv
from PIL import Image
import random
import tables
import cv2
import os
from pathlib import Path
import h5py
import re
import pandas as pd
import itertools
import argparse
from shutil import rmtree, copy
import json
import matplotlib.pyplot as plt
import time
import subprocess

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, GlobalAveragePooling2D, Concatenate, Input
from tensorflow.keras.layers import Resizing, Rescaling, RandomFlip, RandomRotation, RandomZoom, Lambda
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as k
from tensorflow.keras import regularizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.efficientnet import EfficientNetB5, EfficientNetB0
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

start_time = time.time()

# load ids and exclude poor AML slides
root_dir = Path('/media/chia/_note')
df = pd.read_csv(root_dir / 'Bone_marrow_cytology_tiles/Classification_20231202.csv')
slide_list = df['New_Image_ID'].to_list()
'''
# slide_list = [x.name[:6] for x in Path('../Bone_marrow_cytology_tiles/').iterdir() if x.is_dir()]
AML_df = pd.read_csv('../BM_cytology_tile_select/0512_Slide_check_for_classification.csv', header = None, names = ['New_Image_ID', 'Preserve'])
slide_to_exclude = AML_df.loc[AML_df['Preserve'] == 0]['New_Image_ID'].to_list()
slide_list = list(set(slide_list) - set(slide_to_exclude))
'''
random.shuffle(slide_list)
print(len(slide_list))

# predict prob. of good quality of all tiles of all slides and save as csv
log_dir = Path(root_dir / '16tb2/202308_BM_archive_O2/BM_cytology_tile_select/0523_VGG16_20e_0001_then_180e_00001_SGD')
prob_csv_dir = log_dir / 'all_probs'
if not prob_csv_dir.exists():
    os.mkdir(prob_csv_dir)

# slide_list = [TV0001, TV0002, TV0003, ......] (list of strings)
for slide_id in slide_list:
    if (prob_csv_dir/f'{slide_id}.csv').exists() or (prob_csv_dir/f'{slide_id}.lck').exists() or (prob_csv_dir/f'{slide_id}.ok').exists():
        continue
    with open(prob_csv_dir/f'{slide_id}.lck', 'w') as f:
        f.write(str(time.time()))

    k.clear_session()
    epoch_list = [int(x.stem[1:]) for x in log_dir.glob('*.h5') if not 'final' in x.stem]
    model = load_model(log_dir / f'e{max(epoch_list)}.h5')
    print(slide_id)
    prob_df = pd.DataFrame()
    slide_dir = Path(root_dir / f'Bone_marrow_cytology_tiles/{slide_id}_tiles')
    # slide_dir is a directory with tiled images like 'TV0001_32845_138753.png'
    i = 0
    tile_list = []
    img_batch = np.zeros((100, 512, 512, 3))
    for tile_path in slide_dir.iterdir():
        # exclude ambiguous tiles
        if tile_path.name in ['TV0577_68860_175674.png', 'TV0577_67836_167226.png']:
            continue
        try:
            img = cv2.imread(str(tile_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            continue
        img_batch[i%100] = img
        tile_list.append(str(tile_path))
        i += 1
        if i % 100 == 0:
            print(i)
            pred_batch = model.predict_on_batch(img_batch)
            pred_batch = np.squeeze(pred_batch, axis = 1)
            temp = pd.DataFrame.from_records(list(zip(tile_list, pred_batch)), columns=['Path', 'Prob'])
            prob_df = pd.concat([prob_df, temp])
            tile_list = []
            img_batch = np.zeros((100, 512, 512, 3))
    if i % 100 != 0:
        print(i)
        pred_batch = model.predict_on_batch(img_batch[:i%100])
        pred_batch = np.squeeze(pred_batch, axis = 1)
        temp = pd.DataFrame.from_records(list(zip(tile_list, pred_batch)), columns=['Path', 'Prob'])
        prob_df = pd.concat([prob_df, temp])
    prob_df.to_csv(prob_csv_dir / f'{slide_id}.csv', index = False)

    os.rename(prob_csv_dir/f'{slide_id}.lck', prob_csv_dir/f'{slide_id}.ok')

    end_time = time.time()
    