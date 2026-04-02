import mne
import pandas as pd
import numpy as np

def preprocess_ecg(ecg_file, ecg_ann_file, resampling_freq=100, bandpass_filter=(1, 40), notch=60, verbosity=False):

    raw = mne.io.read_raw_edf(ecg_file, preload=True, verbose=verbosity) # import file into mne
    og_fs = raw.info['sfreq']

    raw.pick_channels(['MLII'])
    raw.set_channel_types({'MLII': 'ecg'})
    raw.filter(bandpass_filter[0], bandpass_filter[1], verbose=verbosity, picks='ecg') # 1-40 Hz bandpass to preserve useful ECG signal
    raw.notch_filter(notch, verbose=verbosity, picks='ecg') # remove electrical interference
    raw.resample(resampling_freq, verbose=verbosity) # resample to 100 Hz, especially useful if combining datasets of different sampling rates

    ann_samples = ecg_ann_file.sample # grab R peak samples from .atr file
    ann_classes = ecg_ann_file.symbol # grab labels from .atr file
    
    ann_times = ann_samples / og_fs # obtain sampling frequency-agnostic times
    new_ann_samples = (ann_times * raw.info['sfreq']).astype(int) # reconvert back to samples, but this time with new sampling frequency
    class_dict = {'+': 0, 'A': 1, 'N': 2, 'V': 3} 
    func = np.vectorize(lambda f: class_dict.get(f, f))
    ann_classes_int = func(ann_classes).astype(int) # encoding classes into integers
    zero_arr = np.zeros(len(ann_samples)).astype(int) # adapting the necessary structure for the events array

    events_arr = np.vstack((new_ann_samples, zero_arr, ann_classes_int)).T # making an (N, 3) events array

    epochs = mne.Epochs(raw, events_arr, tmin=-0.5, tmax=0.5) # creating epochs at each R peak

    return epochs