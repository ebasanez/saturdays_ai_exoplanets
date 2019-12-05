# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:58:22 2019

@author: Miguel
"""

import numpy as np
from pathlib import Path
from os import listdir
from os.path import isfile, join

project_folder = Path("C:\\Users\\Miguel\\Google Drive\\self-learning\\ai_saturdays\\project (1)\\saturdays_ai_exoplanets")
data_folder = project_folder/"data"/"lc_folded"/"unlabelled"/"no_centroid"

# Get a list of all files in the folder
all_files = [f for f in listdir(data_folder) if isfile(data_folder/f)]

# Separate into types of file and include full path
X_files = [data_folder/f for f in all_files if "X_" in f]
y_files = [data_folder/f for f in all_files if "Y_" in f]
z_files = [data_folder/f for f in all_files if "Z_" in f]

# Sort lists of files in place (will result in same order for all three)
X_files.sort()
y_files.sort()
z_files.sort()


x = [f for f in all_files if "X_" in f]
r = [int(f[7:12]) for f in x]
s = list(range(0, 7144, 10))

missing_files = [n for n in s if n not in r]


# Get the arrays and concatenate them
X = np.concatenate([np.load(f) for f in X_files], axis=0)
y = np.concatenate([np.load(f) for f in y_files], axis=0)
z = np.concatenate([np.load(f) for f in z_files], axis=0)

np.save(project_folder/"data"/"X_fold_noCentroid_unlabelled.npy", X)
np.save(project_folder/"data"/"y_fold_noCentroid_unlabelled.npy", y)
np.save(project_folder/"data"/"z_fold_noCentroid_unlabelled.npy", z)

