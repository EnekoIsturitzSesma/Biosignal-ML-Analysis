import mne
import os
import numpy as np

def load_physionet(subject_id, base_path, output_path):
    runs = [4, 8, 12]
    all_epochs = []
    
    for run in runs:
        sub_str, run_str = f"S{subject_id:03d}", f"R{run:02d}"
        file_path = os.path.join(base_path, sub_str, f"{sub_str}{run_str}.edf")
        
        if not os.path.exists(file_path): 
            continue

        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        mne.datasets.eegbci.standardize(raw)
        raw.set_montage('standard_1020')
        raw.filter(8., 30., fir_design='firwin', verbose=False)

        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        
        t_mapping = {}
        for key, value in event_dict.items():
            k_upper = key.upper()
            if k_upper in ['T1', '1', 'LEFT', 'L']: 
                t_mapping[value] = 1
            elif k_upper in ['T2', '2', 'RIGHT', 'R']: 
                t_mapping[value] = 2


        relevant_events = [ev for ev in events if ev[2] in t_mapping]
        
        if len(relevant_events) > 0:
            new_events = np.array([ [ev[0], ev[1], t_mapping[ev[2]]] for ev in relevant_events ])
            
            epochs = mne.Epochs(raw, new_events, event_id={'Left': 1, 'Right': 2}, 
                                tmin=-0.5, tmax=4.0, baseline=(None, 0), preload=True, verbose=False)
            all_epochs.append(epochs)

    if all_epochs:
        subject_data = mne.concatenate_epochs(all_epochs)
        out_name = os.path.join(output_path, f"sub-{subject_id:03d}-epo.fif")
        subject_data.save(out_name, overwrite=True, verbose=False)
        print(f"Subject {subject_id} saved")
    else:
        print(f"Subject {subject_id} finnished with  no valid epochs.")

