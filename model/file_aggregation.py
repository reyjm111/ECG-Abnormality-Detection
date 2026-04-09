import wfdb
from preprocess import preprocess_ecg
import numpy as np
import os
import sys
from collect_files import collect

def file_aggregation(base_dir): # getting all the files

    parent_dir = os.path.abspath("..")
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    ecg_files, record_paths = collect(base_dir) 

    X_epochs = []
    y_epochs = []
    all_groups_epochs = []

    # Go through each ecg file and annotation file to obtain epoch, label, and group data
    for ecg_file, record_path in zip(ecg_files, record_paths): 

        ann_file = wfdb.rdann(record_path, 'atr')

        epochs, labels, groups = preprocess_ecg(ecg_file, ann_file) # preprocess ecg and annotation files

        X_epochs.append(epochs)
        y_epochs.append(labels)
        all_groups_epochs.append(groups)

    X = np.concatenate(X_epochs) # concatenate all epochs
    y = np.concatenate(y_epochs) # concatenate all labels
    groups = np.concatenate(all_groups_epochs) # concatenate all groups

    return X, y, groups