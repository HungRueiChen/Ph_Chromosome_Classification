import numpy as np
import tensorflow as tf
from PIL import Image
import random
import cv2
import os
from pathlib import Path
import pandas as pd
import itertools
import argparse
from shutil import rmtree, copy, copytree
from copy import deepcopy
import json
import matplotlib.pyplot as plt
import subprocess
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats

# arguments
parser = argparse.ArgumentParser(description="Summarize and calculate ensemble results of experiments fulfilling the condition within certain directory", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("directory", help = 'image directory, must contain at least one experiment folder')
parser.add_argument("--condition", type = str, default = '5e', help = 'include experiments whose file names contain this string')
args = parser.parse_args()

parent_dir = Path('../') / args.directory
dest_dir = Path('../0sync/') / f'{args.directory}_{args.condition}'
if dest_dir.exists():
    rmtree(dest_dir)
os.mkdir(dest_dir)
# list models to include manually
m_list = ['vgg16', 'inceptionv3', 'resnet50', 'densenet121', 'mobilenetv3large', 'nasnetmobile', 'convnexttiny', 'efficientnetv2b3']
exp_list = [parent_dir / f'{m}_5e_0005_then_35e_00005_SGD' for m in m_list]
# standard practice
# exp_list = [x for x in parent_dir.iterdir() if (x / 'test_metrics.json').exists()]
# exp_list = [x for x in exp_list if args.condition in x.stem]
assert len(exp_list) > 0, "No experiments match the specified criteria."

classes = ['ALL', 'AML_APL', 'CML', 'Lymphoma_CLL', 'MM']
method_list = ['Tile', 'Patient_weighted_average', 'Patient_weighted_sum', 'Patient_mode']
metric_list = ['accuracy', 'balanced_accuracy', 'f1_macro', 'matthews_CC', 'cohen_kappa', 'auc_micro', 'auc_macro']
best_dict = {'architecture': [], 'optimizer': [], 'freeze_lr': [], 'freeze_epochs': [], 'fine_tune_epochs': [], 'best_epoch': []}
statistics_arr = np.zeros((len(exp_list), len(method_list), len(metric_list)), dtype = float)
pt_preds = {}
pt_gts = {}
for m in method_list:
    os.mkdir(dest_dir / m)
os.mkdir(dest_dir / 'best_models')

for exp_num, exp_dir in enumerate(exp_list):
    
    config = exp_dir.name.split('_')
    best_dict['architecture'].append(config[0])
    best_dict['optimizer'].append(config[6])
    best_dict['freeze_lr'].append('0.'+config[2][1:])
    best_dict['freeze_epochs'].append(config[1])
    best_dict['fine_tune_epochs'].append(config[4])
    
    # store best val acc models
    log_df = pd.read_csv(exp_dir / 'log.csv', index_col = 'epoch')
    best_epoch = log_df['val_accuracy'].idxmax() + 1
    best_dict['best_epoch'].append(best_epoch)
    copytree(exp_dir / f'best_e{best_epoch}', dest_dir / 'best_models' / f'{config[0]}_e{best_epoch}')
    copy(exp_dir / 'Learning_curve.png', dest_dir / 'best_models' / f'{config[0]}_Learning_curve.png')
    
    # iterate over methods for best_methods_for_all_exps.csv and [method]_of_all_exps.csv
    with open(exp_dir / 'test_metrics.json', 'r') as f:
        statistics = json.load(f)
    for method_num, method in enumerate(method_list):
        # copy confusion matrices and roc curves first
        save_dir = dest_dir / method / exp_dir.name
        os.makedirs(save_dir)
        figure_list = [x for x in exp_dir.iterdir() if '.png' in x.name and method in x.name]
        for fig_path in figure_list:
            copy(fig_path, save_dir / fig_path.name)
        
        for metric_num, metric in enumerate(metric_list):
            if metric == 'auc_micro':
                statistics_arr[exp_num][method_num][metric_num] = statistics[method]['roc_data']['micro']['auc']
            elif metric == 'auc_macro':
                statistics_arr[exp_num][method_num][metric_num] = statistics[method]['roc_data']['macro']['auc']
            else:
                statistics_arr[exp_num][method_num][metric_num] = statistics[method][metric]
    
    # iterate over patients for ensembles of patient level methods
    with open(exp_dir / 'test_data.json', 'r') as f:
        data = json.load(f)
    if exp_num == 0:
        pt_num = len(data['pt_level'])
        for k, v in data['pt_level'].items():
            pt_preds[k] = np.zeros((len(exp_list), len(method_list)-1, len(classes)))
            pt_gts[k] = np.array(v['y_true']).argmax()
        
    for patient, patient_dict in data['pt_level'].items():
        for method_num, method in enumerate(['y_pred_weighted_ave', 'y_pred_weighted_sum', 'mode_count']):
            pt_preds[patient][exp_num, method_num] = np.array(patient_dict[method])
    
meta_df = pd.DataFrame(best_dict)
# process statistics_arr into best_methods_for_all_exps.csv
best_methods = np.nanargmax(statistics_arr, axis = 1)
for metric_num, metric in enumerate(metric_list):
    best_dict[f'{metric}_method'] = []
    best_dict[metric] = []
    for exp_num, exp_dir in enumerate(exp_list):
        method_idx = best_methods[exp_num, metric_num]
        best_dict[f'{metric}_method'].append(method_list[method_idx])
        best_dict[metric].append(statistics_arr[exp_num, method_idx, metric_num])
best_df = pd.DataFrame(best_dict)
best_df.to_csv(dest_dir / 'Best_of_all_exps.csv', index = False)
# process statistics_arr into [method]_of_all_exps.csv
for method_num, method in enumerate(method_list):
    stat_df = pd.DataFrame(statistics_arr[:, method_num, :], columns = metric_list)
    stat_df = pd.concat([meta_df, stat_df], axis = 1)
    stat_df.to_csv(dest_dir / method / 'Statistics_of_all_exps.csv', index = False)

# process pt_pred for ensembles
pred_arr = np.zeros((pt_num, len(exp_list), len(method_list)-1, len(classes)), dtype = float)
gt_arr = np.zeros((pt_num), dtype = int)
pid_list = []
cnt = 0
for k, v in pt_preds.items():
    pred_arr[cnt] = v
    gt_arr[cnt] = pt_gts[k]
    pid_list.append(k)
    cnt += 1
ensemble_arr = pred_arr.sum(axis = 1).argmax(axis = 2).transpose() # shape (method, patient)

# copy function calculate_metrics: acc, balanced acc, f1, matthew's, cohen's
def calculate_metrics(y_true, y_pred, classes = classes):
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
    result_dict = {}
    result_dict['architecture'] = 'ensemble'
    result_dict['accuracy'] = acc
    result_dict['balanced_accuracy'] = bal_acc
    result_dict['f1_macro'] = f1_macro
    result_dict['matthews_CC'] = matthews
    result_dict['cohen_kappa'] = cohen
    return result_dict

for method_num, method in enumerate(method_list[1:]):
    stat_df = pd.DataFrame(statistics_arr[:, method_num+1, :5], columns = metric_list[:5])
    stat_df = pd.concat([meta_df['architecture'], stat_df], axis = 1)
    ensemble_dict = calculate_metrics(gt_arr, ensemble_arr[method_num])
    ensemble_df = pd.DataFrame(ensemble_dict, index = [len(exp_list)])
    ensemble_df = pd.concat([stat_df, ensemble_df])
    ensemble_df.to_csv(dest_dir / method / 'ensemble.csv', index = False)
    