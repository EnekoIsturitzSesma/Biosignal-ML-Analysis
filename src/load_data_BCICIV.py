import os
import mne
import numpy as np
import scipy.io

mne.set_log_level('WARNING')

EVENT_IDS = {
    769: (0, 'left_hand'),
    770: (1, 'right_hand'),
    771: (2, 'feet'),
    772: (3, 'tongue'),
}

CHANNELS_DEFAULT = [
    'EEG-0', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6',
    'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-13',
]
CHANNELS_ALL = [
    'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3',
    'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz',
    'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10',
    'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz',
    'EEG-15', 'EEG-16',
]


def _resolve_channels(channels_to_use):
    if channels_to_use is None:
        return CHANNELS_DEFAULT
    if channels_to_use == 'all':
        return CHANNELS_ALL
    return channels_to_use


def _extract_trials(signal, events_pos, t_start, t_end, fs, labels, n_classes):
    n_samples = int((t_end - t_start) * fs)
    offset    = int(t_start * fs)

    X_list, y_list = [], []
    for pos, label in zip(events_pos, labels):
        if label >= n_classes:
            continue
        t0 = pos + offset
        t1 = t0 + n_samples
        if t1 <= signal.shape[1]:
            X_list.append(signal[:, t0:t1])
            y_list.append(label)

    return np.array(X_list), np.array(y_list)


def prepare_motor_imagery_dataset(gdf_file, t_start=2.0, t_end=6.0, channels_to_use=None, n_classes=2):
    raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
    fs  = int(raw.info['sfreq'])

    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    active_events      = {k: v for k, v in EVENT_IDS.items() if v[0] < n_classes}

    for event_id in active_events:
        if str(event_id) not in event_dict:
            raise ValueError(f"Evento {event_id} no encontrado en: {event_dict}")

    channels = _resolve_channels(channels_to_use)
    raw.filter(8, 30, fir_design='firwin', verbose=False)
    signal = raw.get_data(picks=channels)

    X_list, y_list = [], []
    for event_id, (label, _) in active_events.items():
        mne_id       = event_dict[str(event_id)]
        event_positions = events[events[:, 2] == mne_id][:, 0]
        X_cls, y_cls = _extract_trials(
            signal, event_positions,
            t_start, t_end, fs,
            labels=[label] * len(event_positions),
            n_classes=n_classes,
        )
        X_list.append(X_cls)
        y_list.append(y_cls)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    class_info = ' | '.join(
        f"{name}: {(y == lbl).sum()}"
        for lbl, (_, name) in enumerate(active_events.values())
        if lbl < n_classes
    )
    print(f"Loading: {os.path.basename(gdf_file)} | {class_info}")

    return {
        'X': X, 'y': y, 'fs': fs,
        'n_channels': len(channels), 'n_classes': n_classes,
        'info': f"Trials: {len(y)} | {class_info}",
    }


def prepare_motor_imagery_dataset_multiband(gdf_file, t_start=2.0, t_end=6.0, bands=[(8, 12), (13, 30)], channels_to_use=None, n_classes=2):
    raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
    fs  = int(raw.info['sfreq'])

    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    active_events = {k: v for k, v in EVENT_IDS.items() if v[0] < n_classes}

    for event_id in active_events:
        if str(event_id) not in event_dict:
            raise ValueError(f"Evento {event_id} no encontrado en: {event_dict}")

    channels = _resolve_channels(channels_to_use)
    raw.pick(channels)

    trial_indices = {
        label: events[events[:, 2] == event_dict[str(eid)]][:, 0]
        for eid, (label, _) in active_events.items()
    }

    X_all_bands, y = [], []

    for l_freq, h_freq in bands:
        signal = raw.copy().filter(l_freq, h_freq, fir_design='firwin',
                                   verbose=False).get_data()
        X_band, y_band = [], []
        for label, positions in trial_indices.items():
            X_cls, y_cls = _extract_trials(
                signal, positions, t_start, t_end, fs,
                labels=[label] * len(positions), n_classes=n_classes,
            )
            X_band.append(X_cls)
            y_band.append(y_cls)

        X_all_bands.append(np.concatenate(X_band))
        if len(y) == 0:
            y = np.concatenate(y_band)

    X = np.stack(X_all_bands, axis=1)

    print(f"Loading (multiband): {os.path.basename(gdf_file)} | Shape: {X.shape}")

    return {
        'X': X, 'y': y, 'fs': fs,
        'n_channels': len(channels), 'n_bands': len(bands), 'n_classes': n_classes,
        'info': f"Shape: {X.shape} | Trials: {len(y)}",
    }


def prepare_eval_dataset(gdf_file, label_file, t_start=2.0, t_end=6.0, channels_to_use=None, n_classes=2):
    raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
    fs  = int(raw.info['sfreq'])

    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    eval_key = '783'
    if eval_key not in event_dict:
        raise ValueError(
            f"Event 783 not found in {os.path.basename(gdf_file)}. "
            f"Available keys: {list(event_dict.keys())}"
        )

    event_positions = events[events[:, 2] == event_dict[eval_key]][:, 0]

    mat    = scipy.io.loadmat(label_file)
    labels = mat['classlabel'].flatten() - 1  

    if len(event_positions) != len(labels):
        raise ValueError(
            f"Trials in GDF ({len(event_positions)}) "
            f"does not match .mat labels ({len(labels)})."
        )

    channels = _resolve_channels(channels_to_use)
    raw.filter(8, 30, fir_design='firwin', verbose=False)
    signal = raw.get_data(picks=channels)

    X, y = _extract_trials(
        signal, event_positions, t_start, t_end, fs,
        labels=labels, n_classes=n_classes,
    )

    class_info = ' | '.join(
        f"{name}: {(y == lbl).sum()}"
        for lbl, (_, name) in enumerate(EVENT_IDS.values())
        if lbl < n_classes
    )
    print(f"Loading (eval): {os.path.basename(gdf_file)} | {class_info}")

    return {
        'X': X, 'y': y, 'fs': fs,
        'n_channels': len(channels), 'n_classes': n_classes,
        'info': f"Trials: {len(y)} | {class_info}",
    }


def prepare_eval_dataset_multiband(gdf_file, label_file, t_start=2.0, t_end=6.0, bands=[(8, 12), (13, 30)], channels_to_use=None, n_classes=2):
    raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
    fs  = int(raw.info['sfreq'])

    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    eval_key = '783'
    if eval_key not in event_dict:
        raise ValueError(f"Event 783 not found in {os.path.basename(gdf_file)}.")

    event_positions = events[events[:, 2] == event_dict[eval_key]][:, 0]

    mat    = scipy.io.loadmat(label_file)
    labels = mat['classlabel'].flatten() - 1

    channels = _resolve_channels(channels_to_use)
    raw.pick(channels)

    X_all_bands, y = [], []

    for l_freq, h_freq in bands:
        signal = raw.copy().filter(l_freq, h_freq, fir_design='firwin',
                                   verbose=False).get_data()
        X_band, y_band = _extract_trials(
            signal, event_positions, t_start, t_end, fs,
            labels=labels, n_classes=n_classes,
        )
        X_all_bands.append(X_band)
        if len(y) == 0:
            y = y_band

    X = np.stack(X_all_bands, axis=1)

    print(f"Loading (eval multiband): {os.path.basename(gdf_file)} | Shape: {X.shape}")

    return {
        'X': X, 'y': y, 'fs': fs,
        'n_channels': len(channels), 'n_bands': len(bands), 'n_classes': n_classes,
        'info': f"Shape: {X.shape} | Trials: {len(y)}",
    }


def load_all_subjects(data_dir, stage='T', label_dir=None, use_multiband=False, bands=[(8, 12), (13, 30)], channels_to_use=None, n_classes=2):
    if stage == 'E' and label_dir is None:
        raise ValueError("label_dir is necessary when stage='E'.")

    files       = os.listdir(data_dir)
    subject_ids = sorted(set(f[:3] for f in files if f.endswith(f'{stage}.gdf')))

    X_all, y_all, subject_all = [], [], []

    for subj_id in subject_ids:
        gdf_file = os.path.join(data_dir, f'{subj_id}{stage}.gdf')
        if not os.path.exists(gdf_file):
            print(f"File not found: {gdf_file}")
            continue

        try:
            if stage == 'T':
                fn   = prepare_motor_imagery_dataset_multiband if use_multiband \
                       else prepare_motor_imagery_dataset
                kwargs = dict(bands=bands) if use_multiband else {}
                data = fn(gdf_file, channels_to_use=channels_to_use,
                          n_classes=n_classes, **kwargs)

            else:  # stage == 'E'
                label_file = os.path.join(label_dir, f'{subj_id}E.mat')
                if not os.path.exists(label_file):
                    print(f"Label file not found: {label_file}")
                    continue
                fn   = prepare_eval_dataset_multiband if use_multiband \
                       else prepare_eval_dataset
                kwargs = dict(bands=bands) if use_multiband else {}
                data = fn(gdf_file, label_file, channels_to_use=channels_to_use,
                          n_classes=n_classes, **kwargs)

            X_all.append(data['X'])
            y_all.append(data['y'])
            subject_all.extend([subj_id] * len(data['y']))

        except Exception as e:
            print(f"Error in {subj_id}: {e}")

    return {
        'X':           np.concatenate(X_all,    axis=0),
        'y':           np.concatenate(y_all,    axis=0),
        'subject_ids': np.array(subject_all),
        'n_classes':   n_classes,
    }


def load_combined(data_dir, label_dir, n_classes=2, use_multiband=False, bands=[(8, 12), (13, 30)], channels_to_use=None):
    common = dict(
        use_multiband=use_multiband, bands=bands,
        channels_to_use=channels_to_use, n_classes=n_classes,
    )
    data_T = load_all_subjects(data_dir, stage='T', **common)
    data_E = load_all_subjects(data_dir, stage='E', label_dir=label_dir, **common)

    X    = np.concatenate([data_T['X'],           data_E['X']],           axis=0)
    y    = np.concatenate([data_T['y'],           data_E['y']],           axis=0)
    subj = np.concatenate([data_T['subject_ids'], data_E['subject_ids']], axis=0)

    n_per_subj = len(y) // len(np.unique(subj))
    print(f"Combined T+E | Shape: {X.shape} | ~{n_per_subj} trials/subject")

    return X, y, subj