import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

from models.LSTMGait import LSTMGait

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


def trainning_loop(model, train_dl, val_dl, num_classes, epochs=100, lr=0.0005, patience=20):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

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


def train_model_cv(X, y, subjects, epochs=100, lr=0.0003, patience=20):

    models_per_subject = [] 

    num_channels = X.shape[-1]
    num_classes = 3

   # run_name = f"EEGNet_{channels}_{'_'.join(transforms) if transforms else 'raw'}"

   # mlflow.set_experiment('BCI_EEGNet')

    # with mlflow.start_run(run_name=run_name):

    #     mlflow.log_param("epochs", epochs)
    #     mlflow.log_param("lr", lr)
    #     mlflow.log_param("patience", patience)
    #     mlflow.log_param("transforms", transforms)

    logo = LeaveOneGroupOut()
    logo.get_n_splits(groups=subjects)

    test_subject_f1s = []

    for i, (train_index, test_index) in enumerate(logo.split(X, y, subjects)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        mean, std = compute_normalization(X_train)

        X_train = apply_normalization(X_train, mean, std)
        X_test = apply_normalization(X_test, mean, std)

        train_dataset = LSTMGaitDataset(X_train, y_train)
        test_dataset = LSTMGaitDataset(X_test, y_test)

        train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=128, shuffle=False)

        model = LSTMGait(num_channels, num_classes, hidden_size=128, num_layers=2, dropout_rate=0.25)

        trained_model, best_f1 = trainning_loop(model, train_dl, test_dl, num_classes, epochs=epochs, lr=lr, patience=patience)
        test_subject_f1s.append(best_f1)
        models_per_subject.append(trained_model)
        print(f"Fold {i+1}: Val F1 = {best_f1:.4f}")

        subject = subjects[test_index][0]

        # mlflow.log_metric(f'subject_{subject}_train_accuracy', train_subject_accuracy)
        # mlflow.log_metric(f'subject_{subject}_test_accuracy', test_subject_accuracy)
        # mlflow.pytorch.log_model(trained_model, artifact_path=f'model_subject_{subject}')

    mean_acc = np.mean(test_subject_f1s)
    std_acc = np.std(test_subject_f1s)
    print(f"Mean Subject f1-score: {mean_acc:.4f}")
    print(f"Standard Deviation: {std_acc:.4f}")

    # mlflow.log_metric('mean_accuracy', mean_acc)
    # mlflow.log_metric('std_accuracy', std_acc)

    return models_per_subject, test_subject_f1s


