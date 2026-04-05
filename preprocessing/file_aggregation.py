import wfdb
from preprocess import preprocess_ecg
import numpy as np
import os
import sys

def file_aggregation(base_dir):

    parent_dir = os.path.abspath("..")
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from collect_files import collect

    files, record_paths = collect(base_dir)

    X = []
    y = []
    all_groups = []

    for file, record_path in zip(files, record_paths):

        ann = wfdb.rdann(record_path, 'atr')

        epochs, labels, groups = preprocess_ecg(file, ann)

        X.append(epochs)
        y.append(labels)
        all_groups.append(groups)

    X = np.concatenate(X)
    y = np.concatenate(y)
    groups = np.concatenate(all_groups)

    return X, y, groups