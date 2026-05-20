import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from src.preprocess import laplacian_filter, normalize_trial, apply_ea_loso, apply_ea_loso_multiband, euclidean_alignment
from models.EEGNet import EEGNet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42



def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def apply_max_norm(model, max_val=1.0):
    for name, param in model.named_parameters():
        if 'weight' in name and param.ndim > 1:
            if any(k in name for k in ('TemporalConv', 'DepthSpatialConv', 'FC')):
                param.data.copy_(torch.renorm(param.data, p=2, dim=0, maxnorm=max_val))


def init_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def make_loader(X, y, transforms, augment, batch_size=32, shuffle=True):
    ds = EEGDataset(X, y, transforms=transforms, augment=augment)
    g  = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=4,
        generator=g                if shuffle else None,
        worker_init_fn=seed_worker if shuffle else None,
        pin_memory=True,
    )


def build_model(channels, samples, num_classes, f1=8, D=2, dropout=0.4):
    model = EEGNet(channels, samples, num_classes, f1=f1, D=D, dropout_rate=dropout)
    model.apply(init_xavier)
    return model


def val_split_stratified(X, y, val_size=0.15):
    idx = np.arange(len(y))
    tr_idx, val_idx = train_test_split(
        idx, test_size=val_size, stratify=y, random_state=SEED
    )
    return tr_idx, val_idx



class EEGDataset(Dataset):

    def __init__(self, X, y, transforms=None, augment=False):
        self.X          = X
        self.y          = y
        self.transforms = transforms or []
        self.augment    = augment

        if 'multiband' in self.transforms or 'mu_band' in self.transforms:
            self.channels = self.X.shape[2]
        else:
            self.channels = self.X.shape[1]

        if 'multiband' in self.transforms:
            assert self.X.ndim == 4, (
                f"Transform 'multiband' requires shape (trials, bands, channels, samples), "
                f"received {self.X.shape}. Load data with use_multiband=True."
            )

        if 'laplacian' in self.transforms:
            if self.channels == 22:
                self.ch_lap   = [7, 11]
                self.c3_neigh = [1, 6, 8, 13]
                self.c4_neigh = [5, 10, 12, 17]
            else:
                self.ch_lap   = [3, 7]
                self.c3_neigh = [0, 2, 4, 9]
                self.c4_neigh = [1, 6, 8, 10]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()

        if 'multiband' in self.transforms and 'laplacian' in self.transforms:
            processed = []
            for b in range(x.shape[0]):
                bd = laplacian_filter(
                    x[b][np.newaxis], self.ch_lap,
                    [self.c3_neigh, self.c4_neigh]
                ).squeeze(0)
                processed.append(bd)
            x = np.concatenate(processed, axis=0)
        else:
            if 'mu_band' in self.transforms:
                x = x[0]
            if 'multiband' in self.transforms and 'laplacian' not in self.transforms:
                bands, ch, samp = x.shape
                x = x.reshape(bands * ch, samp)
            if 'laplacian' in self.transforms:
                x = laplacian_filter(
                    x[np.newaxis], self.ch_lap,
                    [self.c3_neigh, self.c4_neigh]
                ).squeeze(0)

        if self.augment:
            x = self._augment(x)

        x = normalize_trial(x)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )

    def _augment(self, x):
        if np.random.rand() < 0.3:
            x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
        if np.random.rand() < 0.3:
            x = x * np.random.uniform(0.92, 1.08)
        if np.random.rand() < 0.2:
            x = np.roll(x, np.random.randint(-3, 3), axis=-1)
        if np.random.rand() < 0.2:
            ch = np.random.randint(0, x.shape[0])
            x  = x.copy()
            x[ch] = 0
        return x



def training_loop(model, train_dl, val_dl=None, epochs=300, lr=0.0005, patience=40, subject=None, log_prefix=''):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    use_val = val_dl is not None

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max' if use_val else 'min', 
        factor=0.5,
        patience=patience // 2,
        min_lr=1e-6,
    )

    model.to(DEVICE)

    use_val     = val_dl is not None
    best_metric = 0.0 if use_val else float('inf')
    best_state  = None
    counter     = 0

    for epoch in (bar := tqdm(range(epochs), desc=f'S{subject}', leave=False)):

        model.train()
        t_loss = t_correct = 0
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out  = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            apply_max_norm(model)
            t_loss    += loss.item() * bx.size(0)
            t_correct += (out.argmax(1) == by).sum().item()

        t_loss /= len(train_dl.dataset)
        t_acc   = t_correct / len(train_dl.dataset)

        mlflow.log_metric(f'{log_prefix}train_loss_s{subject}', t_loss, step=epoch)
        mlflow.log_metric(f'{log_prefix}train_acc_s{subject}',  t_acc,  step=epoch)

        if use_val:
            val_acc = evaluate(model, val_dl)
            mlflow.log_metric(f'{log_prefix}val_acc_s{subject}', val_acc, step=epoch)
            bar.set_postfix(train=f'{t_acc:.3f}', val=f'{val_acc:.3f}')
            improved = val_acc > best_metric
            current  = val_acc
        else:
            bar.set_postfix(loss=f'{t_loss:.4f}', acc=f'{t_acc:.3f}')
            improved = t_loss < best_metric
            current  = t_loss

        scheduler.step(current)


        if improved:
            best_metric = current
            best_state  = model.state_dict()
            counter     = 0
        else:
            counter += 1
            if counter >= patience:
                bar.write(
                    f'Early stop epoch {epoch} | '
                    f'best {"val_acc" if use_val else "train_loss"}: {best_metric:.4f}'
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metric


@torch.no_grad()
def evaluate(model, dl):
    model.to(DEVICE).eval()
    correct = sum(
        (model(bx.to(DEVICE)).argmax(1) == by.to(DEVICE)).sum().item()
        for bx, by in dl
    )
    return correct / len(dl.dataset)



def run_loso(X, y, subjects, cfg, num_classes, run_name, val_size=0.15):
    transforms = cfg['transforms']
    augment    = cfg['augment']
    lr         = cfg.get('lr',       0.0003)
    epochs     = cfg.get('epochs',   300)
    patience   = cfg.get('patience', 40)

    if X.ndim == 3:
        _, channels, samples = X.shape
    else:
        _, _, channels, samples = X.shape

    if 'multiband' in transforms:
        channels *= 2
        X = apply_ea_loso_multiband(X, subjects)
    else:
        X = apply_ea_loso(X, subjects)

    mlflow.set_experiment('BCI_EEGNet_LOSO')

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**cfg, 'num_classes': num_classes, 'val_size': val_size})

        accs = {}
        for fold, (tr_idx, te_idx) in enumerate(LeaveOneGroupOut().split(X, y, subjects)):

            subject = subjects[te_idx][0]

            tr_tr_idx, tr_val_idx = val_split_stratified(
                X[tr_idx], y[tr_idx], val_size=val_size
            )
            X_tr  = X[tr_idx][tr_tr_idx]
            y_tr  = y[tr_idx][tr_tr_idx]
            X_val = X[tr_idx][tr_val_idx]
            y_val = y[tr_idx][tr_val_idx]

            train_dl = make_loader(X_tr,      y_tr,      transforms, augment)
            val_dl   = make_loader(X_val,     y_val,     transforms, False, shuffle=False)
            test_dl  = make_loader(X[te_idx], y[te_idx], transforms, False, shuffle=False)

            model, best_val = training_loop(build_model(channels, samples, num_classes), train_dl, val_dl=val_dl, epochs=epochs, lr=lr, patience=patience, subject=subject,)
            test_acc = evaluate(model, test_dl)
            accs[subject] = test_acc

            print(f'  Fold {fold+1:02d} | Subject {subject} | '
                  f'Train {len(y_tr)} | Val {len(y_val)} | '
                  f'Best Val {best_val:.4f} | Test {test_acc:.4f}')
            mlflow.log_metric(f's{subject}_val_acc',  best_val)
            mlflow.log_metric(f's{subject}_test_acc', test_acc)

        mean_acc = np.mean(list(accs.values()))
        std_acc  = np.std(list(accs.values()))
        print(f'  → Mean: {mean_acc:.4f} ± {std_acc:.4f}\n')
        mlflow.log_metric('mean_acc', mean_acc)
        mlflow.log_metric('std_acc',  std_acc)

    return {'mean': mean_acc, 'std': std_acc, 'per_subject': accs}


def run_within_subject_T_to_E(X_train, y_train, subjects_train, X_eval,  y_eval,  subjects_eval, cfg, num_classes, run_name, val_size=0.20):
    transforms = cfg['transforms']
    augment    = cfg['augment']
    lr         = cfg.get('lr',       0.0003)
    epochs     = cfg.get('epochs',   300)
    patience   = cfg.get('patience', 40)

    if X_train.ndim == 3:
        _, channels, samples = X_train.shape
    else:
        _, _, channels, samples = X_train.shape

    if 'multiband' in transforms:
        channels *= 2

    mlflow.set_experiment('BCI_EEGNet_within')

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**cfg, 'num_classes': num_classes, 'val_size': val_size})

        accs = {}
        for subject in np.unique(subjects_train):
            mask = subjects_train == subject
            X_s_raw = X_train[mask].copy()
            
            if X_s_raw.ndim == 4:
                X_s = apply_ea_loso_multiband(X_s_raw, np.zeros(len(X_s_raw)))
            else:
                X_s = euclidean_alignment(X_s_raw)

            y_s  = y_train[mask]

            tr_idx, val_idx = val_split_stratified(X_s, y_s, val_size=val_size)
            X_tr,  y_tr  = X_s[tr_idx],  y_s[tr_idx]
            X_val, y_val = X_s[val_idx], y_s[val_idx]

            te_mask = subjects_eval == subject
            X_te_raw = X_eval[te_mask].copy()
    
            if X_te_raw.ndim == 4:
                X_te = apply_ea_loso_multiband(X_te_raw, np.zeros(len(X_te_raw)))
            else:
                X_te = euclidean_alignment(X_te_raw)
            y_te = y_eval[te_mask]

            train_dl = make_loader(X_tr,  y_tr,  transforms, augment)
            val_dl   = make_loader(X_val, y_val, transforms, False, shuffle=False)
            test_dl  = make_loader(X_te,  y_te,  transforms, False, shuffle=False)

            model, best_val = training_loop(build_model(channels, samples, num_classes), train_dl, val_dl=val_dl, epochs=epochs, lr=lr, patience=patience, subject=subject, log_prefix='within_',)
            test_acc = evaluate(model, test_dl)
            accs[subject] = test_acc

            print(f'  Subject {subject} | '
                  f'Train {len(y_tr)} | Val {len(y_val)} | '
                  f'Best Val {best_val:.4f} | Eval(E) {test_acc:.4f}')
            mlflow.log_metric(f's{subject}_val_acc',  best_val)
            mlflow.log_metric(f's{subject}_eval_acc', test_acc)

        mean_acc = np.mean(list(accs.values()))
        std_acc  = np.std(list(accs.values()))
        print(f'  → Mean: {mean_acc:.4f} ± {std_acc:.4f}\n')
        mlflow.log_metric('mean_acc', mean_acc)
        mlflow.log_metric('std_acc',  std_acc)

    return {'mean': mean_acc, 'std': std_acc, 'per_subject': accs}