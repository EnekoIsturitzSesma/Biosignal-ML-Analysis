import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

from src.preprocess import laplacian_filter, normalize_trial
from models.EEGNet import EEGNet

np.random.seed(42)
torch.manual_seed(42)


class EEGDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        self.X = X
        self.y = y
        self.transforms = transforms
        
        if 'multiband' in self.transforms or 'mu_band' in self.transforms:
            self.channels = self.X.shape[1]
        else:
            self.channels = self.X.shape[0]

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

        x = normalize_trial(x)
        x = torch.tensor(x, dtype=torch.float32)
        
        y = torch.tensor(self.y[idx], dtype=torch.long)
        
        return x, y


def apply_max_norm(model, max_val=1.0):
    for name, param in model.named_parameters():
        if 'weight' in name and param.ndim > 1:
            if 'TemporalConv' in name or 'DepthSpatialConv' in name or 'FC' in name:
                param.data.copy_(torch.renorm(param.data, p=2, dim=0, maxnorm=max_val))


def trainning_loop(model, train_dl, val_dl, epochs=100, lr=0.0005, patience=20):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    epoch_bar = tqdm(range(epochs), desc="Training", leave=False)

    for epoch in epoch_bar:

        model.train()
        train_loss = 0.0
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
        train_acc = train_correct / len(train_dl.dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                output = model(batch_x)
                loss = criterion(output, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(output, 1)
                val_correct += (predicted == batch_y).sum().item()

        val_loss /= len(val_dl.dataset)
        val_acc = val_correct / len(val_dl.dataset)
        scheduler.step(val_acc)

        epoch_bar.set_postfix({
            "Train Acc": f"{train_acc:.3f}",
            "Val Acc": f"{val_acc:.3f}",
            "LR": optimizer.param_groups[0]['lr']
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            epoch_bar.write("Early stopping.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_acc


def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_model_cv(X, y, subjects, transforms, epochs=100, lr=0.0003, patience=20):

    if len(X.shape) == 3:
        _, channels, samples = X.shape
    else:
        _, _, channels, samples = X.shape

    if 'multiband' in transforms:
        channels = channels * 2


    logo = LeaveOneGroupOut()
    logo.get_n_splits(groups=subjects)

    test_subject_accuracies = []

    for i, (train_index, test_index) in enumerate(logo.split(X, y, subjects)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_dataset = EEGDataset(X_train, y_train, transforms=transforms)
        test_dataset = EEGDataset(X_test, y_test, transforms=transforms)

        train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = model = EEGNet(channels, samples, 2, f1=32, D=4, dropout_rate=0.4)
        model.apply(init_weights_xavier)

        trained_model, test_subject_accuracy = trainning_loop(model, train_dl, test_dl, epochs=epochs, lr=lr, patience=patience)
        test_subject_accuracies.append(test_subject_accuracy)
        print(f"Fold {i+1}: Test Accuracy: {test_subject_accuracy:.4f}")

    print(f"Mean Subject Accuracy: {np.mean(test_subject_accuracies):.4f}")
    print(f"Standard Deviation: {np.std(test_subject_accuracies):.4f}")

    final_dataset = EEGDataset(X, y, transforms=transforms)

    train_size = int(0.9 * len(final_dataset))
    val_size = len(final_dataset) - train_size

    final_train_subset, final_val_subset = torch.utils.data.random_split(final_dataset,[train_size, val_size])

    final_train_dl = DataLoader(final_train_subset, batch_size=32, shuffle=True)
    final_val_dl = DataLoader(final_val_subset, batch_size=32, shuffle=False)

    model = EEGNet(channels, samples, 2, f1=32, D=4, dropout_rate=0.4)
    model.apply(init_weights_xavier)

    final_model, _ = trainning_loop(model, final_train_dl, final_val_dl, epochs=epochs, lr=lr, patience=patience)

    return final_model, test_subject_accuracies



def evaluate_model(model, X_test, y_test, transforms):

    test_dataset = EEGDataset(X_test, y_test, transforms=transforms)
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total