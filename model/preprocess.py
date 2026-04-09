import mne
import numpy as np
from pathlib import Path

def preprocess_ecg(
    ecg_file, 
    ecg_ann_file, 
    resampling_freq=100, 
    bandpass_filter=(0.1, 40), 
    notch=60, 
    verbosity=False):


    """
    Function: 
        Load ECG .edf file and annotation .atr file. Resample and filter appropriately using MNE. Segment each heartbeat with a total length of t=1 around the R peaks which are in the annotated 
        file. 

    Parameters:
        ecg_ann_file (.edf): EDF file with EKG data (expecting one channel with MLII lead).
        ecg_ann_file (.atr): Annotation file with expert-annotated R peaks and their specific abnormality label.
        resampling_freq (int): Frequency to resample the EKG to, helps with speeding up later training or harmonizing datasets.
        bandpass_filter (tuple): Lowcut and highcut frequencies to bandpass filter the EKG signal.
        notch (int): Notch filter to remove electrical interference.
        verbosity (bool): Whether or not MNE should be verbose

    Outputs:
        epochs (arr): Shape of (N, 1, 101). Explicitly N epochs, 1 channel, 101 time points in a sample.
        labels (arr): Shape of (N, ). Explicitly a 1D array of binary labels.
        groups (arr): Shape of (N, 1): An array with N groups labels.

    """


    name_without_ext = Path(ecg_file).stem  # Returns filename (which should be record)

    raw = mne.io.read_raw_edf(ecg_file, preload=True, verbose=verbosity) # import file into mne
    og_fs = raw.info['sfreq']

    if 'MLII' not in raw.ch_names:
        raise ValueError(f"MLII channel not found in {ecg_file}")
        
    raw.pick(['MLII']) # use a standard single lead
    raw.set_channel_types({'MLII': 'ecg'}) # explicitly mark the channel as ECG
    raw.filter(bandpass_filter[0], bandpass_filter[1], verbose=verbosity, picks='ecg') # 1-40 Hz bandpass to preserve useful ECG signal
    raw.notch_filter(notch, verbose=verbosity, picks='ecg') # remove electrical interference
    raw.resample(resampling_freq, verbose=verbosity) # resample to 100 Hz, especially useful if combining datasets of different sampling rates

    signal = raw.get_data()
    min_sig = signal.min()
    max_sig = signal.max()
    signal_scaled = (signal - min_sig) / (max_sig - min_sig) # min-max scaling each EKG record
    raw_scaled = mne.io.RawArray(signal_scaled, info=raw.info.copy()) # reconverting to Raw object

    ann_samples = np.asarray(ecg_ann_file.sample) # R-peak sample indices
    ann_classes = np.asarray(ecg_ann_file.symbol) # R-peak class label

    exclude_labels = {"+", '"', "~", "[", "]", "|", "!", "x", "/", "f"}
    keep_mask = ~np.isin(ann_classes, list(exclude_labels)) # keep only labels that exist in the class dictionary


    ann_classes = ann_classes[keep_mask] # filter annotation labels down to the selected classes
    ann_samples = ann_samples[keep_mask] # filter annotation sample indices to match the kept labels

    class_dict = { # encode classes into 0 for non-abnormal beats and abnormal beats
        "N": 0,
        "L": 1,
        "R": 1,
        "A": 1,
        "a": 1,
        "J": 1,
        "S": 1,
        "V": 1,
        "E": 1,
        "F": 1,
        "e": 1,
        "j": 1,
        "Q": 1,
    }

    keep_mask2 = np.isin(ann_classes, list(class_dict.keys())) # safety check for only relevant classes

    ann_classes = ann_classes[keep_mask2] # filter annotation labels down to the selected classes
    ann_samples = ann_samples[keep_mask2] # filter annotation sample indices to match the kept labels

    new_ann_samples = np.round((ann_samples / og_fs) * raw.info['sfreq']).astype(int) # convert original annotation samples to sample indices at the new sampling rate
    ann_classes_int = np.array([class_dict[x] for x in ann_classes], dtype=int) # convert kept annotation labels into integer class codes
    zero_arr = np.zeros(len(new_ann_samples), dtype=int) # create the middle column required for the MNE events array structure

    events_arr = np.column_stack((new_ann_samples, zero_arr, ann_classes_int)) # build an (N, 3) MNE events array: [sample, 0, class_id]

    event_id = {'Normal': 0, 'Abnormal': 1}
    
    epochs = mne.Epochs(
        raw_scaled,
        events_arr,
        event_id=event_id,
        tmin=-0.5,
        tmax=0.5,
        preload=True,
        verbose=verbosity,
        baseline=None, 
        on_missing='ignore'
    ) # create 1-second epochs centered around each retained annotation event

    selection = epochs.selection

    epochs = epochs.get_data() # epoch array
    labels = ann_classes_int[selection] # labels array
    groups = np.array([name_without_ext] * len(labels)) # gropus array

    return epochs, labels, groups