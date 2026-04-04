import mne
import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_ecg(ecg_file, ecg_ann_file, resampling_freq=100, bandpass_filter=(1, 40), notch=60, verbosity=False):

    name_without_ext = Path(ecg_file).stem  # Returns 'file' [13]

    raw = mne.io.read_raw_edf(ecg_file, preload=True, verbose=verbosity) # import file into mne
    og_fs = raw.info['sfreq']

    raw.pick(['MLII']) # keep the MLII lead only
    raw.set_channel_types({'MLII': 'ecg'}) # explicitly mark the channel as ECG
    raw.filter(bandpass_filter[0], bandpass_filter[1], verbose=verbosity, picks='ecg') # 1-40 Hz bandpass to preserve useful ECG signal
    raw.notch_filter(notch, verbose=verbosity, picks='ecg') # remove electrical interference
    raw.resample(resampling_freq, verbose=verbosity) # resample to 100 Hz, especially useful if combining datasets of different sampling rates

    ann_samples = np.asarray(ecg_ann_file.sample) # grab annotation sample indices from .atr file
    ann_classes = np.asarray(ecg_ann_file.symbol) # grab annotation labels from .atr file

    class_dict = {'+': 0, 'A': 1, 'N': 2, 'V': 3, 'Q': 4, '|': 5, '~': 6} # encode selected annotation classes into integers
    keep_mask = np.isin(ann_classes, list(class_dict.keys())) # keep only labels that exist in the class dictionary

    ann_classes = ann_classes[keep_mask] # filter annotation labels down to the selected classes
    ann_samples = ann_samples[keep_mask] # filter annotation sample indices to match the kept labels

    new_ann_samples = np.round((ann_samples / og_fs) * raw.info['sfreq']).astype(int) # convert original annotation samples to sample indices at the new sampling rate
    ann_classes_int = np.array([class_dict[x] for x in ann_classes], dtype=int) # convert kept annotation labels into integer class codes
    zero_arr = np.zeros(len(new_ann_samples), dtype=int) # create the middle column required for the MNE events array structure

    events_arr = np.column_stack((new_ann_samples, zero_arr, ann_classes_int)) # build an (N, 3) MNE events array: [sample, 0, class_id]

    present_codes = np.unique(ann_classes_int) # identify which class codes are actually present in this record
    event_id = {k: v for k, v in class_dict.items() if v in present_codes} # only pass event IDs that are actually present in the current events array

    epochs = mne.Epochs(
        raw,
        events_arr,
        event_id=event_id,
        tmin=-0.5,
        tmax=0.5,
        preload=True,
        verbose=verbosity,
        baseline=None
    ) # create 1-second epochs centered around each retained annotation event

    epochs.info['subject_info'] = {'his_id': name_without_ext}

    return epochs