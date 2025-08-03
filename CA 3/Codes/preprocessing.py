import numpy as np


def preprocess_eeg(raw, preprocess_params):
    """
    Preprocess the raw EEG data:
    1. Remove 'Left Ear' and 'corr' channels
    2. Re-reference to the average reference
    3. Apply band-pass filter (0.5 Hz - 100 Hz)
    4. Apply notch filter at 60 Hz
    5. Downsample to 200 Hz
    """
    # 1. Remove 'Left Ear' and 'corr' channels if present
    channels_to_remove = preprocess_params['chs_to_remove']
    if preprocess_params['chs_to_remove'] is not None:
        print('Removing channels: ', preprocess_params['chs_to_remove'])
        raw.drop_channels([ch for ch in channels_to_remove if ch in raw.ch_names])

    # 2. Re-reference to the average reference
    if preprocess_params['ref_channels'] is not None:
        print('Re-referencing to channels: ', preprocess_params['ref_channels'])
        raw.set_eeg_reference(ref_channels=preprocess_params['ref_channels'])

    # 3. Apply a band-pass filter (0.5 Hz to 100 Hz)
    if (preprocess_params['filter_l'] is not None) or (preprocess_params['filter_h'] is not None):
        print('Filtering the signals in range: ', [preprocess_params['filter_l'], preprocess_params['filter_h']])
        raw.filter(l_freq=preprocess_params['filter_l'], h_freq=preprocess_params['filter_h'])

    # 4. Apply a notch filter at 60 Hz to remove power line noise
    if preprocess_params['notch_filter'] is not None:
        print('Notch filtering EEG in: ', preprocess_params['notch_filter'])
        raw.notch_filter(freqs=preprocess_params['notch_filter'])

    # 5. Downsample to 200 Hz
    if preprocess_params['resample'] is not None:
        print('Resampling to: ', preprocess_params['resample'])
        raw.resample(sfreq=preprocess_params['resample'])

    return raw


