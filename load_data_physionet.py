import mne
import numpy as np
import os
from scipy.io import loadmat

def prepare_physionet_dataset(edf_file, exclude_runs=None, t_start=0.5, t_end=4.5):
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    fs = int(raw.info['sfreq'])
    
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    
    if not events.size:
        raise ValueError(f"No events found in: {edf_file}")
    
    t0_id = event_dict.get('T0', None)
    t1_id = event_dict.get('T1', None)
    t2_id = event_dict.get('T2', None)
    
    print(f"Loaded file: {edf_file} | Available events: {event_dict}")
    
    channels_to_use = list(range(58))
    
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


def load_all_subjects_physionet(data_dir):
    X_all = []
    y_all = []
    subject_all = []
    
    subject_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('S') and len(d) == 4])
    
    for subj_id in subject_dirs:
        subject_path = os.path.join(data_dir, subj_id)
        
        if not os.path.isdir(subject_path):
            continue
        
        edf_files = sorted([f for f in os.listdir(subject_path) if f.endswith('.edf')])
        
        edf_files = [f for f in edf_files if not any(f.endswith(f'R0{i}.edf') for i in [1, 2])]
        
        for edf_file in edf_files:
            edf_path = os.path.join(subject_path, edf_file)
            
            try:
                data = prepare_physionet_dataset(edf_path)
                X_all.append(data['X'])
                y_all.append(data['y'])
                subject_all.extend([subj_id] * len(data['y']))
                
            except Exception as e:
                print(f"Error in {subj_id}/{edf_file}: {e}")
    
    if not X_all:
        raise ValueError("No data was loaded. Check the data directory path.")
    
    X_combined = np.concatenate(X_all, axis=0)
    y_combined = np.concatenate(y_all, axis=0)
    
    dataset = {
        'X': X_combined,
        'y': y_combined,
        'subject_ids': np.array(subject_all),
        'fs': 160
    }
    
    return dataset
