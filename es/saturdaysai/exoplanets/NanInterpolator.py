# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:17:35 2019

@author: Miguel
"""

import numpy as np
from pathlib import Path
from os import listdir
from os.path import isfile, join



def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


project_folder = Path("C:\\Users\\Miguel\\Google Drive\\self-learning\\ai_saturdays\\project\\saturdays_ai_exoplanets")
data_folder = project_folder/"data"
X_filename = "X_fold_noCentroid.npy"
y_filename = "y_fold_noCentroid.npy"
z_filename = "z_fold_noCentroid.npy"


# Loading data
Xnans = np.load(data_folder/X_filename)

ynans = np.load(data_folder/y_filename)
znans = np.load(data_folder/z_filename)


nan_number = int(np.isnan(Xnans).sum())
print(f"Number of NaNs in data: {nan_number} out of {Xnans.size}")
print(f"NaN ratio: {nan_number/Xnans.size}")


# Interpolation of NaNs
Xnonans = []
ynonans = []
znonans = []

for i, lc_nans in enumerate(Xnans):
    #print(i)
    lc_glob = lc_nans[:2049]
    lc_loc  = lc_nans[2049:]
    
    nans_g, f_g = nan_helper(lc_glob)
    nans_l, f_l = nan_helper(lc_loc)
    #print(nans_g.sum(), nans_l.sum())
    if (nans_g.sum() > 2049/2) and (nans_l.sum() > 257/2):
        continue
    
    lc_glob[nans_g] = np.interp(f_g(nans_g), f_g(~nans_g), lc_glob[~nans_g])
    lc_loc[nans_l]  = np.interp(f_l(nans_l), f_l(~nans_l), lc_loc[~nans_l])
    
    Xnonans.append(np.concatenate([lc_glob, lc_loc]))
    ynonans.append(ynans[i])
    znonans.append(znans[i])
    
Xnonans = np.stack(Xnonans)
ynonans = np.stack(ynonans)
znonans = np.stack(znonans)

np.save(project_folder/"data"/"X_fold_noCentroid_noNans.npy", Xnonans)
np.save(project_folder/"data"/"y_fold_noCentroid_noNans.npy", ynonans)
np.save(project_folder/"data"/"z_fold_noCentroid_noNans.npy", znonans)



"""
 
y= array([1, 1, 1, NaN, NaN, 2, 2, NaN, 0])
>>>
>>> nans, x= nan_helper(y)
>>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
>>>
>>> print y.round(2)
"""