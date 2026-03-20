import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import f1_score
from tqdm import tqdm
import gc
import json
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

from models.LSTMGait import LSTMGait
from src.load_data_gait import load_trial

np.random.seed(42)
torch.manual_seed(42)

class LSTMGaitDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]  


def compute_normalization(X):

   # X: (windows, samples, features) 
   X_reshaped = X.reshape(-1, X.shape[-1])
   mean = X_reshaped.mean(axis=0)
   std = X_reshaped.std(axis=0)

   return mean, std


def apply_normalization(X, mean, std):

    return (X - mean) / std


def training_loop(model, train_dl, val_dl, num_classes, epochs=100, lr=0.0005, patience=20):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_f1 = 0 
    best_model_state = None
    patience_counter = 0

    epoch_bar = tqdm(range(epochs), desc="Training", leave=False)

    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        train_total = 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, num_classes), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * y.numel()
            train_total += y.numel()

        model.eval()
        val_preds_all = []
        val_true_all = []

        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)

                output = model(x)
                
                preds = output.argmax(dim=2).cpu().numpy().flatten()
                true = y.cpu().numpy().flatten()

                val_preds_all.extend(preds)
                val_true_all.extend(true)

        val_f1 = f1_score(val_true_all, val_preds_all, average='macro')
        
        scheduler.step(val_f1)

        epoch_bar.set_postfix({
            "Val F1": f"{val_f1:.3f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            epoch_bar.write(f"Early stopping in epoch {epoch}. Best F1: {best_val_f1:.4f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_f1



def train_model_cv(X, y, subjects, epochs=100, lr=0.0003, patience=20, out_dir="checkpoints"):

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        done_subjects = {e['subject'] for e in summary}

    else:
        summary = []
        done_subjects = set()

    num_channels = X.shape[-1]
    num_classes = len(np.unique(y))  

    logo = LeaveOneGroupOut()
    test_subject_f1s = []
    models_per_subject = []

    for i, (trainval_index, test_index) in enumerate(logo.split(X, y, subjects)):

        subject = str(subjects[test_index][0])
        if subject in done_subjects:
            ckpt =torch.load(os.path.join(out_dir, f"model_{subject}.pt"), map_location='cpu')
            model = LSTMGait(num_channels, num_classes, hidden_size=128, num_layers=2, dropout_rate=0.25)
            model.load_state_dict(ckpt['state_dict'])
            models_per_subject.append(model)
            test_subject_f1s.append(ckpt['test_f1'])
            continue


        X_trainval, X_test = X[trainval_index], X[test_index]
        y_trainval, y_test = y[trainval_index], y[test_index]
        subjects_trainval = subjects[trainval_index]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_idx, val_idx = next(gss.split(X_trainval, y_trainval, subjects_trainval))

        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        mean, std = compute_normalization(X_train)
        X_train = apply_normalization(X_train, mean, std)
        X_val   = apply_normalization(X_val,   mean, std)
        X_test  = apply_normalization(X_test,  mean, std)

        train_dl = DataLoader(LSTMGaitDataset(X_train, y_train), batch_size=128, shuffle=True)
        val_dl   = DataLoader(LSTMGaitDataset(X_val,   y_val),   batch_size=128, shuffle=False)
        test_dl  = DataLoader(LSTMGaitDataset(X_test,  y_test),  batch_size=128, shuffle=False)

        model = LSTMGait(num_channels, num_classes, hidden_size=128, num_layers=2, dropout_rate=0.25)

        trained_model, best_val_f1 = training_loop(model, train_dl, val_dl, num_classes, epochs=epochs, lr=lr, patience=patience)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model.eval()
        test_preds, test_true = [], []
        with torch.no_grad():
            for x, yb in test_dl:
                x = x.to(device)
                preds = trained_model(x).argmax(dim=2).cpu().numpy().flatten()
                test_preds.extend(preds)
                test_true.extend(yb.numpy().flatten())

        test_f1 = f1_score(test_true, test_preds, average='macro')

        ckpt_path = os.path.join(out_dir, f"model_{subject}.pt")
        torch.save({
            'subject': subject,
            'fold': i,
            'state_dict': trained_model.state_dict(),
            'val_f1': best_val_f1,
            'test_f1': test_f1,
            'norm_mean': mean,
            'norm_std': std,
            'num_channels': num_channels,
            'num_classes': num_classes,
        }, ckpt_path)

        summary.append({
            'subject': subject,
            'fold': i,
            'val_f1': round(best_val_f1, 6),
            'test_f1': round(test_f1, 6),
            'path': ckpt_path
        })

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        test_subject_f1s.append(test_f1)
        models_per_subject.append(trained_model)

        print(f"Fold {i+1} | Subject {subject} | Val F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f}")

        del train_dl, val_dl, test_dl

        del X_train, X_val, X_test, y_train, y_val, y_test
        del X_trainval, y_trainval

        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nMean Test F1: {np.mean(test_subject_f1s):.4f} ± {np.std(test_subject_f1s):.4f}")
    return models_per_subject, test_subject_f1s


def load_model(subject, out_dir="checkpoints/processed"):
    ckpt = torch.load(
        os.path.join(out_dir, f"model_{subject}.pt"),
        map_location='cpu',
        weights_only=False
    )
    model = LSTMGait(
        ckpt['num_channels'],
        ckpt['num_classes'],
        hidden_size=128,
        num_layers=2,
        dropout_rate=0.25
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['norm_mean'], ckpt['norm_std']



def predict_trial(base_path, trial_name, subject, process="preprocessed", window_size=100, stride=25, out_dir="checkpoints/processed"):

    trial = load_trial(base_path, trial_name)
    trial_metadata = trial['metadata']

    if process == "preprocessed":
        X_trial = trial['data_processed']
        X_raw = pd.DataFrame(X_trial)
    elif process == "raw":
        X_trial = trial['data_raw']
        X_raw = pd.concat(
            [df.add_prefix(f"{key}_") for key, df in X_trial.items()], axis=1
        )

    X_clean = X_raw.dropna(how='any').to_numpy()
    n_samples = X_clean.shape[0]


    y_true = np.zeros(n_samples, dtype=np.int64)
    for start, end in trial_metadata['leftGaitEvents']:
        y_true[start:min(end, n_samples)] = 1
    for start, end in trial_metadata['rightGaitEvents']:
        y_true[start:min(end, n_samples)] = 2

    model, norm_mean, norm_std = load_model(subject, out_dir)

    windows = []
    window_starts = list(range(0, n_samples - window_size + 1, stride))
    for start in window_starts:
        windows.append(X_clean[start:start + window_size])
    X_wins = np.stack(windows).astype(np.float32) 

    X_wins = apply_normalization(X_wins, norm_mean, norm_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        output = model(torch.from_numpy(X_wins).to(device))
        preds  = output.argmax(dim=2).cpu().numpy() 

    num_classes = output.shape[2]
    votes = np.zeros((n_samples, num_classes), dtype=np.int32)

    for i, start in enumerate(window_starts):
        for t in range(window_size):
            votes[start + t, preds[i, t]] += 1

    covered = votes.sum(axis=1) > 0
    y_pred = np.zeros(n_samples, dtype=np.int64)
    y_pred[covered] = votes[covered].argmax(axis=1)

    last_valid = window_starts[-1] + window_size
    if last_valid < n_samples:
        y_pred[last_valid:] = y_pred[last_valid - 1]

    return y_pred, y_true, X_clean