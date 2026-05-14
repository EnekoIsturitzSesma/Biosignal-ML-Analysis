import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
import random
import mlflow
import mlflow.pytorch

import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

from src.preprocess import laplacian_filter, normalize_trial
from models.EEGNet import EEGNet

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark     = False 


class EEGDataset(Dataset):
    def __init__(self, X, y, transforms=None, augment=False):
        self.X = X
        self.y = y
        self.transforms = transforms
        self.augment = augment
        
        if 'multiband' in self.transforms or 'mu_band' in self.transforms:
            self.channels = self.X.shape[2]
        else:
            self.channels = self.X.shape[1]

        if 'laplacian' in self.transforms:
            if self.channels == 22:
                self.channels_laplace = [7,11] # C3 and C4
                self.c3_neighbours = [1,6,8,13]
                self.c4_neighbours = [5,10,12,17]
            else:
                self.channels_laplace = [3,7] # C3 and C4
                self.c3_neighbours = [0,2,4,9]
                self.c4_neighbours = [1,6,8,10]
            

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        if 'multiband' in self.transforms and 'laplacian' in self.transforms:
            bands, channels, samples = x.shape

            processed_bands = []
            for band in range(bands):
                band_data = x[band]
                band_data = np.expand_dims(band_data, axis=0)
                band_data = laplacian_filter(band_data, self.channels_laplace, [self.c3_neighbours, self.c4_neighbours])
                band_data = band_data.squeeze(0)
                processed_bands.append(band_data)

            x = np.concatenate(processed_bands, axis=0)

        else:
            if 'mu_band' in self.transforms:
                x = x[0]
            
            if 'laplacian' in self.transforms:
                x = np.expand_dims(x, axis=0)
                x = laplacian_filter(x, self.channels_laplace, [self.c3_neighbours, self.c4_neighbours])
                x = x.squeeze(0)

        if self.augment:
            x = self._augment(x.numpy() if isinstance(x, torch.Tensor) else x)

        x = normalize_trial(x)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

    def _augment(self, x):
        if np.random.rand() < 0.3:
            x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
        if np.random.rand() < 0.3:
            x = x * np.random.uniform(0.92, 1.08)
        if np.random.rand() < 0.2:
            shift = np.random.randint(-3, 3)
            x = np.roll(x, shift, axis=-1)
        if np.random.rand() < 0.2:
            ch = np.random.randint(0, x.shape[0])
            x = x.copy()
            x[ch] = 0
        return x


def apply_max_norm(model, max_val=1.0):
    for name, param in model.named_parameters():
        if 'weight' in name and param.ndim > 1:
            if 'TemporalConv' in name or 'DepthSpatialConv' in name or 'FC' in name:
                param.data.copy_(torch.renorm(param.data, p=2, dim=0, maxnorm=max_val))


def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def training_loop(model, train_dl, epochs=100, lr=0.0005, patience=20, subject=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
 
    best_train_loss = float('inf')
    best_train_acc  = 0.0
    best_model_state = None
    patience_counter = 0
 
    epoch_bar = tqdm(range(epochs), desc="Training", leave=False)
 
    for epoch in epoch_bar:
        model.train()
        train_loss    = 0.0
        train_correct = 0
 
        for batch_x, batch_y in train_dl:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
 
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            apply_max_norm(model, max_val=1.0)
 
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(output, 1)
            train_correct += (predicted == batch_y).sum().item()
 
        train_loss /= len(train_dl.dataset)
        train_acc   = train_correct / len(train_dl.dataset)

        mlflow.log_metric(f"train_loss_{subject}", train_loss, step=epoch)
        mlflow.log_metric(f"train_acc_{subject}",  train_acc,  step=epoch)
 
        scheduler.step()
 
        epoch_bar.set_postfix({
            "Train Loss": f"{train_loss:.4f}",
            "Train Acc":  f"{train_acc:.3f}",
            "LR":         f"{optimizer.param_groups[0]['lr']:.6f}"
        })
 
        if train_loss < best_train_loss:
            best_train_loss  = train_loss
            best_train_acc   = train_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
 
        if patience_counter >= patience:
            epoch_bar.write(f"Early stopping in epoch {epoch}. "
                            f"Best train loss: {best_train_loss:.4f}")
            break
 
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
 
    return model, best_train_acc
 
 
def evaluate(model, test_dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
 
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in test_dl:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            _, predicted = torch.max(model(batch_x), 1)
            correct += (predicted == batch_y).sum().item()
 
    return correct / len(test_dl.dataset)
 
 
def train_model_cv(X, y, subjects, transforms, epochs=100, lr=0.0003, patience=20, augment=False):

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
 
    if len(X.shape) == 3:
        _, channels, samples = X.shape
    else:
        _, _, channels, samples = X.shape
 
    if 'multiband' in transforms:
        channels = channels * 2
 
    models_per_subject = []

    run_name = f"EEGNet_{channels}_{'_'.join(transforms)}_{'aug' if augment else 'noaug'}"
 
    mlflow.set_experiment('BCI_EEGNet')
 
    with mlflow.start_run(run_name=run_name):
 
        mlflow.log_param("epochs",     epochs)
        mlflow.log_param("lr",         lr)
        mlflow.log_param("patience",   patience)
        mlflow.log_param("transforms", transforms)
        mlflow.log_param("augment", augment)
 
        logo = LeaveOneGroupOut()
        test_subject_accuracies = []
 
        for i, (train_index, test_index) in enumerate(logo.split(X, y, subjects)):

            subject = subjects[test_index][0]
 
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
 
            train_dataset = EEGDataset(X_train, y_train, transforms=transforms, augment=augment)
            test_dataset  = EEGDataset(X_test,  y_test,  transforms=transforms, augment=False)

            g = torch.Generator()
            g.manual_seed(42)
 
            train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, generator=g, worker_init_fn=seed_worker, pin_memory=True)
            test_dl  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
 
            model = EEGNet(channels, samples, 2, f1=32, D=4, dropout_rate=0.4)
            model.apply(init_weights_xavier)
 
            trained_model, train_acc = training_loop(model, train_dl, epochs=epochs, lr=lr, patience=patience, subject=subject)
 
            test_acc = evaluate(trained_model, test_dl)
 
            test_subject_accuracies.append(test_acc)
            models_per_subject.append(trained_model)
 
            subject = subjects[test_index][0]
            print(f"Fold {i+1} | Subject {subject} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
 
            mlflow.log_metric(f'subject_{subject}_train_accuracy', train_acc)
            mlflow.log_metric(f'subject_{subject}_test_accuracy',  test_acc)
            mlflow.pytorch.log_model(trained_model, artifact_path=f'model_subject_{subject}')
            
 
        mean_acc = np.mean(test_subject_accuracies)
        std_acc  = np.std(test_subject_accuracies)
        print(f"\nMean Subject Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
 
        mlflow.log_metric('mean_accuracy', mean_acc)
        mlflow.log_metric('std_accuracy',  std_acc)
 
    return models_per_subject, test_subject_accuracies



def training_loop_with_val(model, train_dl, val_dl, epochs=100, lr=0.0005, patience=20, subject=None):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_acc     = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in (bar := tqdm(range(epochs), desc="Training", leave=False)):
        model.train()
        for batch_x, batch_y in train_dl:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            apply_max_norm(model, max_val=1.0)

        scheduler.step()

        val_acc = evaluate(model, val_dl)   

        mlflow.log_metric(f"val_acc_{subject}", val_acc, step=epoch)
        bar.set_postfix({"Val Acc": f"{val_acc:.3f}"})

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                bar.write(f"Early stopping época {epoch}. Best val: {best_val_acc:.4f}")
                break

    model.load_state_dict(best_model_state)
    return model, best_val_acc




def train_model_within_subject(X, y, subjects, transforms, epochs=100, lr=0.0003, patience=20, augment=False):
    
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    if len(X.shape) == 3:
        _, channels, samples = X.shape
    else:
        _, _, channels, samples = X.shape

    if 'multiband' in transforms:
        channels = channels * 2

    models_per_subject     = []
    test_subject_accuracies = []

    run_name = f"EEGNet_within_{channels}_{'_'.join(transforms)}_{'aug' if augment else 'noaug'}"
    mlflow.set_experiment('BCI_EEGNet_within')

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "epochs": epochs, "lr": lr,
            "patience": patience, "transforms": transforms, "augment": augment,
        })

        for subject in np.unique(subjects):

            mask       = subjects == subject
            X_s, y_s   = X[mask], y[mask]

            n          = len(X_s)
            idx        = np.random.permutation(n)
            train_end  = int(0.60 * n)
            val_end    = int(0.80 * n)

            train_idx  = idx[:train_end]
            val_idx    = idx[train_end:val_end]
            test_idx   = idx[val_end:]

            X_train, y_train = X_s[train_idx], y_s[train_idx]
            X_val,   y_val   = X_s[val_idx],   y_s[val_idx]
            X_test,  y_test  = X_s[test_idx],  y_s[test_idx]

            train_dataset = EEGDataset(X_train, y_train, transforms=transforms, augment=augment)
            val_dataset   = EEGDataset(X_val,   y_val,   transforms=transforms, augment=False)
            test_dataset  = EEGDataset(X_test,  y_test,  transforms=transforms, augment=False)

            g = torch.Generator()
            g.manual_seed(42)

            train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                  num_workers=4, generator=g, worker_init_fn=seed_worker, pin_memory=True)
            val_dl   = DataLoader(val_dataset,   batch_size=32, shuffle=False,
                                  num_workers=4, pin_memory=True)
            test_dl  = DataLoader(test_dataset,  batch_size=32, shuffle=False,
                                  num_workers=4, pin_memory=True)

            model = EEGNet(channels, samples, 2, f1=32, D=4, dropout_rate=0.4)
            model.apply(init_weights_xavier)

            trained_model, train_acc = training_loop_with_val(
                model, train_dl, val_dl,
                epochs=epochs, lr=lr, patience=patience, subject=subject
            )

            test_acc = evaluate(trained_model, test_dl)
            test_subject_accuracies.append(test_acc)
            models_per_subject.append(trained_model)

            print(f"Subject {subject} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")

            mlflow.log_metric(f'subject_{subject}_test_accuracy', test_acc)

        mean_acc = np.mean(test_subject_accuracies)
        std_acc  = np.std(test_subject_accuracies)
        print(f"\nMean Within-Subject Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        mlflow.log_metric('mean_accuracy', mean_acc)
        mlflow.log_metric('std_accuracy',  std_acc)

    return models_per_subject, test_subject_accuracies