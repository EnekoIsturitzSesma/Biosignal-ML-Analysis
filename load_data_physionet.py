import mne
import numpy as np
import os

def prepare_physionet_dataset(edf_file, exclude_runs=None, t_start=0.5, t_end=4.5):
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    fs = int(raw.info['sfreq'])
    
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    
    if not events.size:
        raise ValueError(f"No events found in: {edf_file}")
    
    t1_id = event_dict.get('T1', None)
    t2_id = event_dict.get('T2', None)
        
    channels_to_use = ["C3..","C4..","Cz..","Fc3.","Fc4.","Cp3.","Cp4.","P3..","P4..","Fz.."] 
    
    n_samples_trial = int((t_end - t_start) * fs)
    start_idx = int(t_start * fs)
    
    signal = raw.get_data(picks=channels_to_use)
    
    X_list = []
    y_list = []
    
    if t1_id is not None:
        t1_events = events[events[:, 2] == t1_id]
        for event_pos in t1_events[:, 0]:
            trial_start = event_pos + start_idx
            trial_end = trial_start + n_samples_trial
            
            if trial_end <= signal.shape[1]:
                X_list.append(signal[:, trial_start:trial_end].T)
                y_list.append(0)
    
    if t2_id is not None:
        t2_events = events[events[:, 2] == t2_id]
        for event_pos in t2_events[:, 0]:
            trial_start = event_pos + start_idx
            trial_end = trial_start + n_samples_trial
            
            if trial_end <= signal.shape[1]:
                X_list.append(signal[:, trial_start:trial_end].T)
                y_list.append(1)
    
    if not X_list:
        raise ValueError(f"No valid trials extracted from: {edf_file}")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    data = {
        'X': X,
        'y': y,
        'fs': fs,
        'n_channels': len(channels_to_use),
        'info': f"Trials: {len(y)} | T1: {(y==0).sum()} | T2: {(y==1).sum()}"
    }
    
    return data
