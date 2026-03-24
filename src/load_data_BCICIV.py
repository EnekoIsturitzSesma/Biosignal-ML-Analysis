import mne
import numpy as np
import os

mne.set_log_level('WARNING')

def prepare_motor_imagery_dataset(gdf_file, t_start=2.0, t_end=6.0, channels_to_use=None):
    raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
    fs = int(raw.info['sfreq'])
    
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    left_mne_id = event_dict.get('769', None) 
    right_mne_id = event_dict.get('770', None) 
    
    if left_mne_id is None or right_mne_id is None:
        raise ValueError(f"No events found in: {event_dict}")
    
    left_events = events[events[:, 2] == left_mne_id]
    right_events = events[events[:, 2] == right_mne_id]
    
    print(f"Loaded file: {gdf_file}")
    
    if channels_to_use is None:
        channels_to_use = ['EEG-0', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7','EEG-C4', 'EEG-8', 'EEG-9', 'EEG-13']
    elif channels_to_use == 'all':
        channels_to_use = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']
    
    n_samples_trial = int((t_end - t_start) * fs)
    start_idx = int(t_start * fs)
    
    raw.filter(7, 35, fir_design='firwin', verbose=False)
    

    signal = raw.get_data(picks=channels_to_use)
    
    X_list = []
    y_list = []
    
    for event_pos in left_events[:, 0]:
        trial_start = event_pos + start_idx
        trial_end = trial_start + n_samples_trial
        
        if trial_end <= signal.shape[1]:
            X_list.append(signal[:, trial_start:trial_end])
            y_list.append(0)
    
    for event_pos in right_events[:, 0]:
        trial_start = event_pos + start_idx
        trial_end = trial_start + n_samples_trial
        
        if trial_end <= signal.shape[1]:
            X_list.append(signal[:, trial_start:trial_end])
            y_list.append(1)
    
    X = np.array(X_list) 
    y = np.array(y_list) 
    
    data = {
        'X': X,
        'y': y,
        'fs': fs,
        'n_channels': len(channels_to_use),
        'info': f"Trials: {len(y)} | Left: {(y==0).sum()} | Right: {(y==1).sum()}"
    }
    
    return data


def prepare_motor_imagery_dataset_multiband(gdf_file, t_start=2.0, t_end=6.0, bands=[(8, 12), (13, 30)], channels_to_use=None):
    raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
    fs = int(raw.info['sfreq'])
    
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    left_mne_id = event_dict.get('769', None) 
    right_mne_id = event_dict.get('770', None) 
    
    if left_mne_id is None or right_mne_id is None:
        raise ValueError(f"No events found in: {event_dict}")
        
    left_events = events[events[:, 2] == left_mne_id]
    right_events = events[events[:, 2] == right_mne_id]

    if channels_to_use is None:
        channels_to_use = ['EEG-0', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7','EEG-C4', 'EEG-8', 'EEG-9', 'EEG-13']
    elif channels_to_use == 'all':
        channels_to_use = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']
    raw.pick(channels_to_use)
    
    n_samples_trial = int((t_end - t_start) * fs)
    start_idx = int(t_start * fs)
    
    X_all_bands = [] 
    y = [] 
    
    for (l_freq, h_freq) in bands:
        raw_band = raw.copy().filter(l_freq, h_freq, fir_design='firwin', verbose=False)
        signal = raw_band.get_data() 
        
        X_band = []
        y_band = []
        
        for event_pos in left_events[:, 0]:
            trial_start = event_pos + start_idx
            trial_end = trial_start + n_samples_trial
            if trial_end <= signal.shape[1]:
                X_band.append(signal[:, trial_start:trial_end])
                y_band.append(0)
                
        for event_pos in right_events[:, 0]:
            trial_start = event_pos + start_idx
            trial_end = trial_start + n_samples_trial
            if trial_end <= signal.shape[1]:
                X_band.append(signal[:, trial_start:trial_end])
                y_band.append(1)
        
        X_all_bands.append(np.array(X_band))
        
        if len(y) == 0:
            y = np.array(y_band)
            
    X = np.stack(X_all_bands, axis=1)
    
    data = {
        'X': X,
        'y': y,
        'fs': fs,
        'n_channels': len(channels_to_use),
        'n_bands': len(bands),
        'info': f"Trials: {len(y)} | Left: {(y==0).sum()} | Right: {(y==1).sum()} | Shape: {X.shape}"
    }
    
    return data


def load_all_subjects(data_dir, stage='T', use_multiband=False, bands=[(8, 12), (13, 30)], channels_to_use=None):
    files = os.listdir(data_dir)
    subject_ids = sorted(list(set([f[0:3] for f in files if f.endswith(f'{stage}.gdf')])))
    
    X_all = []
    y_all = []
    subject_all = []
    
    for subj_id in subject_ids:
        gdf_file = f"{data_dir}/{subj_id}{stage}.gdf"
        
        if not os.path.exists(gdf_file):
            print(f"File not found: {gdf_file}")
            continue
        
        try:
            if use_multiband:
                data = prepare_motor_imagery_dataset_multiband(gdf_file, bands=bands, channels_to_use=channels_to_use)
            else:
                data = prepare_motor_imagery_dataset(gdf_file, channels_to_use=channels_to_use)
            X_all.append(data['X'])
            y_all.append(data['y'])
            subject_all.extend([subj_id] * len(data['y']))
            
        except Exception as e:
            print(f"Error in {subj_id}: {e}")
    
    X_combined = np.concatenate(X_all, axis=0)
    y_combined = np.concatenate(y_all, axis=0)

    dataset = {
        'X': X_combined,
        'y': y_combined,
        'subject_ids': np.array(subject_all)
    }
    
    return dataset