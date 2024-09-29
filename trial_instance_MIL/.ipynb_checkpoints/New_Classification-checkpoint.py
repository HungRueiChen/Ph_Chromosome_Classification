import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import csv
from PIL import Image
import random
import cv2
import os
from pathlib import Path
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
from tensorflow.keras.optimizers.legacy import RMSprop, Adam, SGD
from tensorflow.keras import backend as k
from tensorflow.keras import regularizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.convnext import ConvNeXtTiny
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from tensorflow_addons.optimizers import RectifiedAdam
from AlexNet.alexnet import AlexNet

from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

# arguments
parser = argparse.ArgumentParser(description="Train and Test with Metrics on Given Directory", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("directory", help = 'image directory, must contain 3 subdirectories: train, val, test')
parser.add_argument("--mode", choices = ['initiate', 'resume'], default = 'initiate', help = 'start from epoch 1 or resume from last stored model')
parser.add_argument("--architecture", choices = ['alexnet', 'vgg16', 'inceptionv3', 'resnet50', 'densenet121', 'xception', 'mobilenetv3large', 'inceptionresnetv2', 'nasnetmobile', 'convnexttiny', 'efficientnetv2b3'], default = 'vgg16', help = 'CNN backbbone architecture')
parser.add_argument("--optimizer", choices = ['SGD', 'Adam', 'RMSprop', 'rAdam'], default = 'SGD', help = 'optimizer')
parser.add_argument("--batch_size", type = int, default = 8, help = 'batch size')
parser.add_argument("--freeze_lr", type = str, default = '0.001', help = 'learning rate when base model is freezed at first, fine tune lr will be 1/10 of freeze lr')
parser.add_argument("--freeze_epochs", type = int, default = 10, help = 'epochs for freezing stage')
parser.add_argument("--fine_tune_epochs", type = int, default = 90, help = 'epochs for fine tune stage')
parser.add_argument("--high_wt", type = float, default = 1, help = 'multiply by x the class weights that > 1')
args = parser.parse_args()

# json encoder for special data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# define variables
mother_dir = Path('/media/chia/_note/16tb2/Ph_Chromosome_Classification') / args.directory
training_dir = mother_dir / 'training'
validation_dir = mother_dir / 'validation'
test_dir = mother_dir / 'test'

train_ds = image_dataset_from_directory(training_dir, labels = 'inferred', label_mode = 'categorical', batch_size = args.batch_size, image_size = (512, 512), follow_links = True)
val_ds = image_dataset_from_directory(validation_dir, labels = 'inferred', label_mode = 'categorical', batch_size = args.batch_size, image_size = (512, 512), follow_links = True)

class_weight = {}
classes = np.concatenate([y for x, y in train_ds], axis = 0).argmax(axis = 1)
for i, w in enumerate(compute_class_weight('balanced', classes = [0, 1], y = classes)):
    class_weight[i] = w * args.high_wt if w > 1 else w 
print(class_weight)

freeze_lr = float(args.freeze_lr)
fine_tune_lr = freeze_lr * 0.1
fc_node = 256

fr_lr_str = args.freeze_lr.replace('.', '')
tu_lr_str = '0' + fr_lr_str
log_dir = Path(mother_dir / f'{args.architecture}_{args.freeze_epochs}e_{fr_lr_str}_then_{args.fine_tune_epochs}e_{tu_lr_str}_{args.optimizer}')

# model establishment & training configuration
if args.mode == 'initiate':
    if log_dir.exists():
        rmtree(log_dir)
    os.mkdir(log_dir)
    
    latest_epoch = 0
    best_val_acc = 0
    freeze_epochs = args.freeze_epochs
    fine_tune_epochs = args.fine_tune_epochs
    
    if args.architecture == 'alexnet':
        base_model = AlexNet(include_top = False)
        preprocess_fn = lambda x: x / 127.5 -1
        resize_side = 227
    elif args.architecture == 'vgg16':
        base_model = VGG16(include_top = False, weights = 'imagenet', pooling = 'avg')
        preprocess_fn = tf.keras.applications.vgg16.preprocess_input
        resize_side = 224
    elif args.architecture == 'inceptionv3':
        base_model = InceptionV3(include_top = False, weights = 'imagenet', pooling = 'avg')
        preprocess_fn = tf.keras.applications.inception_v3.preprocess_input
        resize_side = 229
    elif args.architecture == 'resnet50':
        base_model = ResNet50(include_top = False, weights = 'imagenet', pooling = 'avg')
        preprocess_fn = tf.keras.applications.resnet.preprocess_input
        resize_side = 224
    elif args.architecture == 'densenet121':
        base_model = DenseNet121(include_top = False, weights = 'imagenet', pooling = 'avg')
        preprocess_fn = tf.keras.applications.densenet.preprocess_input
        resize_side = 224
    elif args.architecture == 'xception':
        base_model = Xception(include_top = False, weights = 'imagenet', pooling = 'avg')
        preprocess_fn = tf.keras.applications.xception.preprocess_input
        resize_side = 299
    elif args.architecture == 'mobilenetv3large':
        base_model = MobileNetV3Large(include_top = False, weights = 'imagenet', pooling = 'avg', alpha = 0.75)
        preprocess_fn = tf.keras.applications.mobilenet_v3.preprocess_input
        resize_side = 224
    elif args.architecture == 'inceptionresnetv2':
        base_model = InceptionResNetV2(include_top = False, weights = 'imagenet', pooling = 'avg')
        preprocess_fn = tf.keras.applications.inception_resnet_v2.preprocess_input
        resize_side = 299
    elif args.architecture == 'nasnetmobile':
        base_model = NASNetMobile(include_top = False, weights = 'imagenet', pooling = 'avg', input_shape = (224, 224, 3))
        preprocess_fn = tf.keras.applications.nasnet.preprocess_input
        resize_side = 224
    elif args.architecture == 'convnexttiny':
        base_model = ConvNeXtTiny(include_top = False, weights = 'imagenet', pooling = 'avg')
        preprocess_fn = tf.keras.applications.convnext.preprocess_input
        resize_side = 224
    elif args.architecture == 'efficientnetv2b3':
        base_model = EfficientNetV2B3(include_top = False, weights = 'imagenet', pooling = 'avg')
        preprocess_fn = tf.keras.applications.efficientnet_v2.preprocess_input
        resize_side = 224
    else:
        raise

    input_layer = Input(shape = (512, 512, 3))
    x = RandomFlip("horizontal")(input_layer)
    x = RandomRotation((-1, 1), fill_mode = 'constant', fill_value = 255)(x)
    x = RandomZoom((-0.1, 0.1), fill_mode = 'constant', fill_value = 255)(x)

    x = Resizing(resize_side, resize_side)(x)
    x = preprocess_fn(x)
    x = base_model(x, training = False)

    x = Dense(fc_node, activation = 'relu', name = 'fc')(x)
    x = Dropout(0.3, name = 'do')(x)
    output_layer = Dense(2, activation = 'softmax', name = 'cl')(x)
    model = Model(inputs = input_layer, outputs = output_layer)

elif args.mode == 'resume':
    ''' TODO: save as tf, read two epoch list (.h5, tf) and resume at the larger of the two maxes. '''
    tf_epoch_list = [int(x.stem[12:]) for x in log_dir.iterdir() if x.is_dir() and 'checkpoint' in x.stem]
    latest_epoch = max(tf_epoch_list) if tf_epoch_list else 0
    assert latest_epoch > 0, f"No previous models found in {log_dir}, unable to resume."
        
    freeze_epochs = max(0, (args.freeze_epochs - latest_epoch))
    fine_tune_epochs = min(args.fine_tune_epochs, (args.fine_tune_epochs + args.freeze_epochs - latest_epoch))
    
    print(f'Loading model: e{latest_epoch}')
    model = load_model(log_dir / f'checkpoint_e{latest_epoch}', compile = True)
    
    # truncate log.csv to current epoch
    log_df = pd.read_csv(log_dir / 'log.csv', index_col = 'epoch')
    log_df = log_df.truncate(before = 0, after = latest_epoch-1)
    best_val_acc = log_df['val_accuracy'].max()
    log_df.to_csv(log_dir / 'log.csv', mode = 'w')

# checkpoint = ModelCheckpoint(filepath = str(log_dir)+'/e{epoch}.h5', monitor = 'val_accuracy', save_best_only = True)
csv_logger = CSVLogger(log_dir / 'log.csv', append = True)

# Training

# stage 1: train top layer only
if freeze_epochs > 0:
    # initialize if no model is loaded
    if freeze_epochs == args.freeze_epochs:
        base_model.trainable = False

        if args.optimizer == 'SGD':
            optimizer_config = SGD(learning_rate = freeze_lr)
        elif args.optimizer == 'Adam':
            optimizer_config = Adam(learning_rate = freeze_lr)
        elif args.optimizer == 'RMSprop':
            optimizer_config = RMSprop(learning_rate = freeze_lr)
        elif args.optimizer == 'rAdam':
            optimizer_config = RectifiedAdam(learning_rate = freeze_lr)

        model.compile(loss = 'categorical_crossentropy', optimizer = optimizer_config, metrics=['accuracy'])
    model.summary()
    
    for epoch in range(latest_epoch, args.freeze_epochs):
        current_time = time.time()
        history = model.fit(train_ds, validation_data = val_ds, epochs = epoch + 1, initial_epoch = epoch, class_weight = class_weight, callbacks = [csv_logger])
        if history.history['val_accuracy'][0] > best_val_acc:
            model.save(log_dir / f'best_e{epoch + 1}', save_format = 'tf')
            best_val_acc = history.history['val_accuracy'][0]
        latest_epoch = epoch + 1

# stage 2: unfreeze all layers
if fine_tune_epochs > 0:
    # initialize if no model is loaded
    if fine_tune_epochs == args.fine_tune_epochs:
        model.trainable = True

        if args.optimizer == 'SGD':
            optimizer_config = SGD(learning_rate = fine_tune_lr)
        elif args.optimizer == 'Adam':
            optimizer_config = Adam(learning_rate = fine_tune_lr)
        elif args.optimizer == 'RMSprop':
            optimizer_confic = RMSprop(learning_rate = fine_tune_lr)
        elif args.optimizer == 'rAdam':
            optimizer_config = RectifiedAdam(learning_rate = fine_tune_lr)

        model.compile(loss = 'categorical_crossentropy', optimizer = optimizer_config, metrics=['accuracy'])
    model.summary()
    
    for epoch in range(latest_epoch, (args.fine_tune_epochs + args.freeze_epochs)):
        current_time = time.time()
        history = model.fit(train_ds, validation_data = val_ds, epochs = epoch + 1, initial_epoch = epoch, class_weight = class_weight, callbacks = [csv_logger])
        if history.history['val_accuracy'][0] > best_val_acc:
            model.save(log_dir / f'best_e{epoch + 1}', save_format = 'tf')
            best_val_acc = history.history['val_accuracy'][0]
        latest_epoch = epoch + 1
        
