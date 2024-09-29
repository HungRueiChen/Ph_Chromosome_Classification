import numpy as np
from PIL import Image
import random
import cv2
import os
from pathlib import Path
import pandas as pd
import itertools
import argparse
from shutil import rmtree, copy
from copy import deepcopy
import json

root_dir = Path('/media/chia/_note')
parent_dir = root_dir / '16tb2/Ph_Chromosome_Classification/trial_MIL_dataset'

# load cohort data from json
with open(root_dir / '16tb2/Ph_Chromosome_Classification/trial_instance_MIL_cohort.json', 'r') as f:
    temp = json.load(f)
cohort = deepcopy(temp)

for group in ['training', 'validation', 'test']:
    for label in ['Ph-', 'Ph+']:
        cohort[group][label] = {'ids': cohort[group][label], 'tiles': []}
        for slide_id in temp[group][label]:
            df = pd.read_csv(root_dir / f'16tb2/202308_BM_archive_O2/BM_cytology_tile_select/0523_VGG16_20e_0001_then_180e_00001_SGD/all_probs/{slide_id}.csv')
            tile_folder = Path(root_dir / f'Bone_marrow_cytology_tiles/{slide_id}_tiles/')
            os.makedirs(parent_dir / group / label, exist_ok = True)
            
            if slide_id == 'TV0577':
                excluded_list = [f'../Bone_marrow_cytology_tiles/TV0577_tiles/{x}' for x in ['TV0577_68860_175674.png', 'TV0577_67836_167226.png']]
                df = df.loc[~df['Path'].isin(excluded_list)]
            
            
            if df.loc[df['Prob'] >= 0.8].shape[0] < 200:
                # fewer than 200 tiles have good quality score > 0.8, include tiles whose scores are between 0.5 and 0.8
                chosen1 = df.loc[df['Prob'] >= 0.8]
                remaining = 200 - len(chosen1)
                chosen2 = df.loc[(df['Prob'] < 0.8) & (df['Prob'] >= 0.5)].sample(n = remaining)
                chosen = pd.concat([chosen1, chosen2])
            else:
                # more than 200 tiles have good quality score > 0.8, randomly select 200 from them as ROIs
                chosen = df.loc[df['Prob'] >= 0.8].sample(n = 200)
            
            for i, row in chosen.iterrows():
                tile = Path(row['Path']).name
                try:
                    copy(tile_folder / tile, parent_dir / group / label / tile)
                    cohort[group][label]['tiles'].append(tile)
                except Exception as e:
                    print(tile, e)
# store included ROIs in another json file
with open(root_dir / '16tb2/Ph_Chromosome_Classification/trial_instance_MIL_tiles.json', 'w') as f:
    json.dump(cohort, f)