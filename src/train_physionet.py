import numpy as np
import mne
import os
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut


def get_loso_data(processed_dir, test_subject_idx, file_list):
    X_train, y_train = [], []
    X_test, y_test = None, None
    
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(processed_dir, file_name)
        epochs = mne.read_epochs(file_path, preload=True, verbose=False)
        
        X = epochs.get_data() 
        y = epochs.events[:, -1] - 1
        
        if i == test_subject_idx:
            X_test, y_test = X, y
        else:
            X_train.append(X)
            y_train.append(y)
            
    return np.concatenate(X_train), np.concatenate(y_train), X_test, y_test


def train_physionet(data_path, data, pipeline, cv_strategy, param_grid):

    X_train, y_train, X_test, y_test = get_loso_data(data_path, 0, data)

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv_strategy, scoring='f1_macro', n_jobs=-1)

    
