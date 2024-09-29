#!/usr/bin/env python
# coding: utf-8

# ## Import Module

# In[1]:


import numpy as np
import csv
from pathlib import Path
import pandas as pd
import json
import random


# In[5]:


# save the json file and start copying to directories
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


# In[25]:
# which WSIs to include as the total cohort

# Classification_20230909.csv has two columns: 'New_Image_ID': ['TV0001', ...] and 'Recheck': ['AML', ...]
root_dir = Path('/media/chia/_note/')
df = pd.read_csv(root_dir / '16tb2/Ph_Chromosome_Classification/merged_subtypes.csv')
original_df = pd.read_csv(root_dir / 'Bone_marrow_cytology_tiles/Classification_20231202.csv')
original_inclusion = original_df.loc[original_df['Exclusion'] == 0]
included_df = df.loc[df['Image_ID'].isin(original_inclusion['New_Image_ID'])]
# print(f'Total: {len(included_df)}')


# In[26]:
# split WSIs into training, validation, and test sets in the proportions of 7:1:2

all_group = {'training': dict(), 'validation': dict(), 'test': dict()}
check_duplicate = {}

def create_cohort(cls_list, included_df, all_group):
    s_list = included_df.loc[included_df['assistant_chrom'] == cls_list]['Image_ID'].to_list()
    s_list = list(set(s_list))
    check_duplicate[cls_list] = s_list
    random.shuffle(s_list)
    cutoff1 = int(len(s_list) * 0.6)
    delta = (len(s_list) - cutoff1) / 2.0
    cutoff2 = cutoff1 + int(np.rint(delta))
    all_group['training'][cls_list] = s_list[:cutoff1]
    all_group['validation'][cls_list] = s_list[cutoff1:cutoff2]
    all_group['test'][cls_list] = s_list[cutoff2:]
    
for cls_list in ['Ph+', 'Ph-']:
    create_cohort(cls_list, included_df, all_group)
assert set(check_duplicate['Ph+']) & set(check_duplicate['Ph-']) == set()

# In[27]:
# store in json format for cohort copying

with open(root_dir / '16tb2/Ph_Chromosome_Classification/trial_instance_MIL_cohort.json', 'w') as f:
    json.dump(all_group, f, cls = NumpyEncoder)