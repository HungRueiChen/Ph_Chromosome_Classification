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
import roc_utils as ru
import matplotlib.pyplot as plt
from scipy import stats

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
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow_addons.optimizers import RectifiedAdam

from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

# arguments
parser = argparse.ArgumentParser(description="Test the Best Model on Given Directory with Metrics ", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("directory", help = 'image directory, must contain at least one model')
parser.add_argument("log", help = 'experiment log directory, must exist under argument [directory]')
parser.add_argument("--subset", choices = ['training', 'validation', 'test'], default = 'test', help = 'image subset to test on')
parser.add_argument("--architecture", choices = ['vgg16', 'inceptionv3', 'resnet50', 'densenet121', 'xception', 'mobilenetv3large', 'inceptionresnetv2', 'nasnetmobile', 'convnexttiny', 'efficientnetv2b3'], default = 'vgg16', help = 'CNN backbbone architecture')
parser.add_argument("--optimizer", choices = ['SGD', 'Adam', 'RMSprop', 'rAdam'], default = 'SGD', help = 'optimizer')
parser.add_argument("--batch_size", type = int, default = 8, help = 'batch size')
parser.add_argument("--freeze_lr", type = str, default = '0.001', help = 'learning rate when base model is freezed at first, fine tune lr will be 1/10 of freeze lr')
parser.add_argument("--freeze_epochs", type = int, default = 10, help = 'epochs for freezing stage')
parser.add_argument("--fine_tune_epochs", type = int, default = 90, help = 'epochs for fine tune stage')
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
classes = ['Ph+', 'Ph-']
mother_dir = Path('/media/chia/_note/16tb2/Ph_Chromosome_Classification') / args.directory
log_dir = mother_dir / args.log
test_dir = mother_dir / args.subset
'''
freeze_lr = float(args.freeze_lr)
fine_tune_lr = freeze_lr * 0.1
fr_lr_str = args.freeze_lr.replace('.', '')
tu_lr_str = '0' + fr_lr_str
log_dir = Path(mother_dir / f'{args.architecture}_{args.freeze_epochs}e_{fr_lr_str}_then_{args.fine_tune_epochs}e_{tu_lr_str}_{args.optimizer}')
'''
if not log_dir.exists():
    raise

# learning curve
if (log_dir / 'log.csv').exists():
    log_df = pd.read_csv(log_dir / 'log.csv', index_col = 'epoch')
    loss = log_df['loss'].to_list()
    val_loss = log_df['val_loss'].to_list()
    acc = log_df['accuracy'].to_list()
    val_acc = log_df['val_accuracy'].to_list()
    epochs_range = range(1, len(loss)+1)

    learning_curve = plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right', fontsize = 10, prop={'family':'serif'})
    plt.title('Training and Validation Loss', fontsize = 15, fontfamily = 'serif')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right', fontsize = 10, prop={'family':'serif'})
    plt.title('Training and Validation Accuracy', fontsize = 15, fontfamily = 'serif')

    learning_curve.suptitle(f'Learning Curves: {log_dir.name}')
    learning_curve.savefig(log_dir / 'Learning_curve.png', bbox_inches = 'tight')

# Test Section
# Define functions
def calculate_metrics(y_true, y_pred, f_name, result_dict, classes):
    # initialize
    cm_norm = confusion_matrix(y_true, y_pred, normalize = 'true')
    
    # accuracy = micro f1
    correct = np.equal(y_true, y_pred)
    acc = sum(correct) / len(correct)
    
    # balanced accuracy
    recalls = [cm_norm[x][x] for x in range(len(classes))]
    bal_acc = np.mean(recalls)
    
    # f1 score (macro/micro)
    f1_macro = f1_score(y_true, y_pred, average = 'macro')
    f1_micro = f1_score(y_true, y_pred, average = 'micro') # should = accuracy

    # Matthews correlation coefficient
    matthews = matthews_corrcoef(y_true, y_pred)

    # Cohen's kappa
    cohen = cohen_kappa_score(y_true, y_pred)
    
    # save to dict
    if f_name not in result_dict:
        result_dict[f_name] = {}
    result_dict[f_name]['accuracy'] = acc
    result_dict[f_name]['balanced_accuracy'] = bal_acc
    result_dict[f_name]['f1_macro'] = f1_macro
    result_dict[f_name]['matthews_CC'] = matthews
    result_dict[f_name]['cohen_kappa'] = cohen

def draw_cm(y_true, y_pred, log_dir, best_epoch, f_name, result_dict, classes):
    # initialize
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize = 'true')
    
    fig_cm = plt.figure(figsize = (18, 6))

    for i, mtx in enumerate([cm, cm_norm.round(3)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(mtx, interpolation = 'nearest', vmin = 0, cmap = plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation = 0, fontsize = 10, fontfamily = 'serif')
        plt.yticks(tick_marks, classes, fontsize = 10, fontfamily = 'serif')
        thresh = mtx.max() / 2.
        for i, j in itertools.product(range(mtx.shape[0]), range(mtx.shape[1])):
            plt.text(j, i, mtx[i, j], horizontalalignment = "center", fontsize = 15, fontfamily = 'serif', color = "white" if mtx[i, j] > thresh else "black")

        plt.ylabel('Groundtruth', fontsize = 12, fontfamily = 'serif')
        plt.xlabel('Prediction', fontsize = 12, fontfamily = 'serif')
        subtitle = 'Normalized Confusion Matrix' if i else 'Confusion Matrix'

    # fig_cm.suptitle(f'{f_name}: {log_dir.name}_e{best_epoch}', fontsize = 12, fontfamily = 'serif')
    fig_cm.savefig(log_dir / f'{f_name}_confusion_matrix.png', dpi = fig_cm.dpi, bbox_inches = 'tight')
    plt.tight_layout()
    plt.close()
    
    if f_name not in result_dict:
        result_dict[f_name] = {}
    result_dict[f_name]['confusion_matrix'] = cm
    result_dict[f_name]['normalized_cm'] = cm_norm
    
def draw_roc_curve(y_true, y_pred, log_dir, best_epoch, f_name, result_dict, classes):
    roc_data = {}

    fig_roc = plt.figure(figsize = (18, 5.6))
    plt.subplot(1, 3, 1)
    colors = ['firebrick', 'orange', 'limegreen', 'deepskyblue', 'deeppink']

    # roc curve as per class
    for i, study_group in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        area = auc(fpr, tpr)
        roc_data[study_group] = {'fpr': fpr, 'tpr': tpr, 'auc': area}

        plt.plot(fpr, tpr, label = f'{study_group}, area = {area:.3f}', color = colors[i], linewidth = 1)

    plt.plot([0, 1], [0, 1], 'k--', lw = 1)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 12, fontfamily = 'serif')
    plt.ylabel('True Positive Rate', fontsize = 12, fontfamily = 'serif')
    plt.legend(loc = "lower right", fontsize = 8, prop={'family':'serif'})

    # roc curves micro
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    area = auc(fpr, tpr)
    roc_data['micro'] = {'fpr': fpr, 'tpr': tpr, 'auc': area}

    # roc curves macro
    fpr_grid = np.linspace(0.0, 1.0, 1001)
    mean_tpr = np.zeros_like(fpr_grid)
    for study_group in classes:
        mean_tpr += np.interp(fpr_grid, roc_data[study_group]['fpr'], roc_data[study_group]['tpr'])
    # average it and compute AUC
    mean_tpr /= len(classes)
    mac_area = auc(fpr_grid, mean_tpr)
    fpr_grid = np.insert(fpr_grid, 0, 0)
    mean_tpr = np.insert(mean_tpr, 0, 0)
    roc_data['macro'] = {'fpr': fpr_grid, 'tpr': mean_tpr, 'auc': mac_area}

    # plot roc micro and macro
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, label = f'micro-average, area = {area:.3f}', color = 'red', linewidth = 1)
    plt.plot(fpr_grid, mean_tpr, label = f'macro-average, area = {mac_area:.3f}', color = 'blue', linewidth = 1)
    plt.plot([0, 1], [0, 1], 'k--', lw = 1)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 12, fontfamily = 'serif')
    plt.ylabel('True Positive Rate', fontsize = 12, fontfamily = 'serif')
    plt.legend(loc = "lower right", fontsize = 8, prop={'family':'serif'})
    
    # bootstrap TI of roc micro
    n_samples = 10000 if 'Patient' in f_name else 1000
    ret_mean = ru.compute_roc_bootstrap(X = y_pred.ravel(), y = y_true.ravel(), pos_label = 1, n_bootstrap = n_samples, return_mean = True)
    tpr_sort = np.sort(ret_mean.tpr_all, axis=0)
    tpr_lower = tpr_sort[int(0.025 * n_samples), :]
    tpr_upper = tpr_sort[int(0.975 * n_samples), :]
    roc_data['micro']['auc_mean'] = ret_mean["auc_mean"]
    roc_data['micro']['auc95_ci'] = ret_mean["auc95_ci"][0]
    roc_data['micro']['auc95_ti'] = ret_mean["auc95_ti"][0]
    roc_data['micro']['auc_std'] = ret_mean["auc_std"]
    
    # plot roc micro with TI
    plt.subplot(1, 3, 3)    
    plt.plot(fpr, tpr, label = f'micro-average, area = {area:.3f}', color = 'red', linewidth = 1, zorder = 3)
    plt.fill_between(ret_mean.fpr, tpr_lower, tpr_upper, color='gray', alpha=.3, label=f'95% interval, area = {ret_mean.auc95_ti[0]}', zorder=2)
    plt.plot([0, 1], [0, 1], 'k--', lw = 1)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 12, fontfamily = 'serif')
    plt.ylabel('True Positive Rate', fontsize = 12, fontfamily = 'serif')
    plt.legend(loc = "lower right", fontsize = 8, prop={'family':'serif'})

    # fig_roc.suptitle(f'{f_name}: {log_dir.name}_e{best_epoch}', fontsize = 15, fontfamily = 'serif')
    fig_roc.savefig(log_dir / f'{f_name}_ROC_curve.png', dpi = fig_roc.dpi, bbox_inches = 'tight')
    plt.close()
    
    if f_name not in result_dict:
        result_dict[f_name] = {}
    result_dict[f_name]['roc_data'] = roc_data
'''
def metrics(y_true_one_hot, y_pred_ori, log_dir, best_epoch, f_name, result_dict, classes = ['ALL', 'AML', 'CML', 'Lymphoma', 'MM']):
    calculate_metrics(y_true_one_hot, y_pred_ori, f_name, result_dict, classes)
    draw_cm(y_true_one_hot, y_pred_ori, log_dir, best_epoch, f_name, result_dict, classes)
    draw_roc_curve(y_true_one_hot, y_pred_ori, log_dir, best_epoch, f_name, result_dict, classes)
'''
# Start testing
test_ds = image_dataset_from_directory(test_dir, batch_size = args.batch_size,
                                       labels = 'inferred', label_mode = 'categorical',
                                       image_size = (512, 512), shuffle = False, follow_links = True)
# choose the best model
if (log_dir / 'log.csv').exists():
    log_df = pd.read_csv(log_dir / 'log.csv', index_col = 'epoch')
    best_epoch = log_df['val_accuracy'].idxmax() + 1
else:
    raise
    
print(f'Loading model: e{best_epoch}')
model = load_model(log_dir / f'best_e{best_epoch}')

test_data = {}

# tile level data
y_true_one_hot = np.concatenate([y for x, y in test_ds], axis = 0)
y_pred_ori = model.predict(test_ds)
test_data['y_true'] = y_true_one_hot
test_data['y_pred'] = y_pred_ori

# patient level data
def weighted_average(y_pred):
    # TODO: sum of squares / sum for axis == 0
    matrix = np.array(y_pred)
    matrix_sum_of_sq = np.square(matrix).sum(axis = 0)
    matrix_sum = matrix.sum(axis = 0)
    result = np.divide(matrix_sum_of_sq, matrix_sum, out = np.zeros_like(matrix_sum), where = (matrix_sum != 0))
    return result

def weighted_sum(y_pred):
    matrix = np.array(y_pred)
    matrix_sum = np.square(matrix).sum(axis = 0)
    result = matrix_sum / matrix_sum.sum(axis = 0, keepdims = 1)
    return result

def mode_count(y_pred):
    matrix = np.array(y_pred)
    pred_labels = matrix.argmax(axis = 1)
    mode_cnt = np.zeros((len(classes)), dtype = int)
    unique, counts = np.unique(pred_labels, return_counts = True)
    for i, label in enumerate(unique):
        mode_cnt[int(label)] = counts[i]
    return mode_cnt

def mode_of_argmax(y_pred):
    mode_cnt = mode_count(y_pred)
    return mode_cnt.argmax()

pt_level_data = {}
x_slidename = [Path(x).name[:6] for x in test_ds.file_paths]
for x_f, y_t, y_p in zip(x_slidename, test_data['y_true'], test_data['y_pred']):
    if x_f not in pt_level_data:
        pt_level_data[x_f] = {'y_true': y_t, 'y_pred': [], 'tiles_num': 0}
    pt_level_data[x_f]['y_pred'].append(y_p)
    pt_level_data[x_f]['tiles_num'] += 1

for k, v in pt_level_data.items():
    v['y_pred_weighted_ave'] = weighted_average(v['y_pred'])
    v['y_pred_weighted_sum'] = weighted_sum(v['y_pred'])
    v['mode_count'] = mode_count(v['y_pred'])
    v['mode'] = mode_of_argmax(v['y_pred'])

test_data['pt_level'] = pt_level_data
    
with open(log_dir / 'test_data.json', 'w') as f:
    json.dump(test_data, f, cls = NumpyEncoder)
    
# calculate metrics
result_dict = {}

# tile level
f_name = f'Tile'
y_true = test_data['y_true'].argmax(axis = 1)
y_pred = test_data['y_pred'].argmax(axis = 1)
calculate_metrics(y_true, y_pred, f_name, result_dict, classes)
draw_cm(y_true, y_pred, log_dir, best_epoch, f_name, result_dict, classes)
draw_roc_curve(test_data['y_true'], test_data['y_pred'], log_dir, best_epoch, f_name, result_dict, classes)

# patient level, weighted average
f_name = f'Patient_weighted_average'
pt_temp = [(v['y_true'], v['y_pred_weighted_ave']) for k, v in test_data['pt_level'].items()]
pt_y_true_wa, pt_y_pred_wa = zip(*pt_temp)
y_true = np.array(pt_y_true_wa).argmax(axis = 1)
y_pred = np.array(pt_y_pred_wa).argmax(axis = 1)
calculate_metrics(y_true, y_pred, f_name, result_dict, classes)
draw_cm(y_true, y_pred, log_dir, best_epoch, f_name, result_dict, classes)
draw_roc_curve(np.array(pt_y_true_wa), np.array(pt_y_pred_wa), log_dir, best_epoch, f_name, result_dict, classes)

# patient level, weighted sum
f_name = f'Patient_weighted_sum'
pt_temp = [(v['y_true'], v['y_pred_weighted_sum']) for k, v in test_data['pt_level'].items()]
pt_y_true_ws, pt_y_pred_ws = zip(*pt_temp)
y_true = np.array(pt_y_true_ws).argmax(axis = 1)
y_pred = np.array(pt_y_pred_ws).argmax(axis = 1)
calculate_metrics(y_true, y_pred, f_name, result_dict, classes)
draw_cm(y_true, y_pred, log_dir, best_epoch, f_name, result_dict, classes)
draw_roc_curve(np.array(pt_y_true_ws), np.array(pt_y_pred_ws), log_dir, best_epoch, f_name, result_dict, classes)
''' omit because metrics (except ROC curves) do not change after normalization as in below
# patient level, mode
f_name = 'Patient_mode'
pt_temp = [(v['y_true'], v['mode']) for k, v in test_data['pt_level'].items()]
pt_y_true_mo, pt_y_pred_mo = zip(*pt_temp)
y_true = np.array(pt_y_true_mo).argmax(axis = 1)
y_pred = np.array(pt_y_pred_mo)
calculate_metrics(y_true, y_pred, f_name, result_dict, classes)
draw_cm(y_true, y_pred, log_dir, best_epoch, f_name, result_dict, classes)
'''
# patient level, mode count for roc curve
f_name = f'Patient_mode'
pt_temp = [(v['y_true'], v['mode_count']) for k, v in test_data['pt_level'].items()]
pt_y_true_mo, pt_y_pred_mo = zip(*pt_temp)
y_true = np.array(pt_y_true_mo).argmax(axis = 1)
y_pred_mo = np.array(pt_y_pred_mo)
y_pred_mo = y_pred_mo / y_pred_mo.sum(axis = 0, keepdims = 1)
y_pred = np.array(pt_y_pred_mo).argmax(axis = 1)
calculate_metrics(y_true, y_pred, f_name, result_dict, classes)
draw_cm(y_true, y_pred, log_dir, best_epoch, f_name, result_dict, classes)
draw_roc_curve(np.array(pt_y_true_mo), y_pred_mo, log_dir, best_epoch, f_name, result_dict, classes)

with open(log_dir / 'test_metrics.json', 'w') as f:
    json.dump(result_dict, f, cls = NumpyEncoder)

method_list = ['Tile', 'Patient_weighted_average', 'Patient_weighted_sum', 'Patient_mode']
metric_list = ['accuracy', 'balanced_accuracy', 'f1_macro', 'matthews_CC', 'cohen_kappa', 'auc_micro', 'auc_macro']
result_arr = np.zeros((len(method_list), len(metric_list)))
for i, method in enumerate(method_list):
    for j, metric in enumerate(metric_list):
        if metric == 'auc_micro':
            result_arr[i][j] = result_dict[method]['roc_data']['micro']['auc']
        elif metric == 'auc_macro':
            result_arr[i][j] = result_dict[method]['roc_data']['macro']['auc']
        else:
            result_arr[i][j] = result_dict[method][metric]
result_df = pd.DataFrame(result_arr, columns = metric_list, index = method_list)
result_df.to_csv(log_dir / 'test_metrics.csv')