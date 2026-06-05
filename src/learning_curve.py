
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                        
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.linalg
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from sktime.transformations.panel.rocket import MiniRocketMultivariate



N_CLASSES = 2          
N_SEEDS   = 3          
RUN_EEGNET = True      


N_TRIALS_LIST = [50, 100, 144, 288, 576, 1152, 2304]

WITHIN_SUBJECT_N = 144 if N_CLASSES == 2 else 288

CSP_CONFIG = dict(
    n_components=4,
    reg='ledoit_wolf',
)
MINIROCKET_CONFIG = dict(
    num_kernels=10_000,
    random_state=42,
    alphas=np.logspace(-3, 3, 10),
)
EEGNET_CONFIG = dict(
    transforms=['base'],   
    augment=False,         
    lr=0.0003,
    epochs=300,            
    patience=30,
    f1=8,
    D=2,
    dropout=0.4,
)

OUTPUT_PATH = "learning_curve.png"




def subsample_stratified(X, y, n, seed=42):
    if n >= len(y):
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    n_per_class = n // len(classes)
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        n_take = min(n_per_class, len(c_idx))
        chosen = rng.choice(c_idx, n_take, replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    rng.shuffle(idx)
    return idx


def euclidean_alignment(X):
    covs = np.array([x @ x.T / x.shape[-1] for x in X])
    R_mean = covs.mean(axis=0)
    R_inv_sqrt = np.linalg.inv(scipy.linalg.sqrtm(R_mean)).real
    return np.array([R_inv_sqrt @ x for x in X])


def zscore_trials(X):
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True) + 1e-8
    return (X - mean) / std




def eval_csp(X_tr, y_tr, X_te, y_te, cfg):
    X_tr = zscore_trials(X_tr.astype(np.float64))
    X_te = zscore_trials(X_te.astype(np.float64))

    csp = CSP(
        n_components=cfg['n_components'],
        reg=cfg['reg'],
        log=True,
    )
    lda = LinearDiscriminantAnalysis(solver='svd')

    X_tr_feat = csp.fit_transform(X_tr, y_tr)
    X_te_feat = csp.transform(X_te)
    lda.fit(X_tr_feat, y_tr)
    return lda.score(X_te_feat, y_te)


def eval_minirocket(X_tr, y_tr, X_te, y_te, cfg):
    X_tr_ea = euclidean_alignment(zscore_trials(X_tr))
    X_te_ea = euclidean_alignment(zscore_trials(X_te))

    transform = MiniRocketMultivariate(
        num_kernels=cfg['num_kernels'],
        random_state=cfg['random_state'],
    )
    transform.fit(X_tr_ea)
    X_tr_feat = transform.transform(X_tr_ea)
    X_te_feat = transform.transform(X_te_ea)

    scaler = StandardScaler()
    X_tr_feat = scaler.fit_transform(X_tr_feat)
    X_te_feat = scaler.transform(X_te_feat)

    clf = RidgeClassifierCV(alphas=cfg['alphas'])
    clf.fit(X_tr_feat, y_tr)
    return clf.score(X_te_feat, y_te)


def eval_eegnet(X_tr, y_tr, X_te, y_te, channels, samples, num_classes, cfg):
    try:
        from train_EEGNet import (
            build_model, make_loader, training_loop,
            evaluate, val_split_stratified, DEVICE,
        )
    except Exception as e:
        raise ImportError(
            f"Can't import train_EEGNet. {e}"
        )

    transforms = cfg.get('transforms', ['base'])
    augment    = cfg.get('augment', False)

    X_tr_ea = euclidean_alignment(zscore_trials(X_tr))
    X_te_ea = euclidean_alignment(zscore_trials(X_te))

    n_val = max(int(0.15 * len(y_tr)), num_classes) 
    tr_idx, val_idx = val_split_stratified(X_tr_ea, y_tr, val_size=n_val / len(y_tr))

    train_dl = make_loader(X_tr_ea[tr_idx],  y_tr[tr_idx],  transforms, augment)
    val_dl   = make_loader(X_tr_ea[val_idx], y_tr[val_idx], transforms, False, shuffle=False)
    test_dl  = make_loader(X_te_ea,          y_te,          transforms, False, shuffle=False)

    model = build_model(
        channels, samples, num_classes,
        f1=cfg.get('f1', 8),
        D=cfg.get('D', 2),
        dropout=cfg.get('dropout', 0.4),
    )
    model, _ = training_loop(
        model, train_dl, val_dl=val_dl,
        epochs=cfg.get('epochs', 300),
        lr=cfg.get('lr', 0.0003),
        patience=cfg.get('patience', 30),
        subject='lc',
    )
    return evaluate(model, test_dl)



def compute_learning_curves(
    X, y, subjects,
    n_trials_list=N_TRIALS_LIST,
    n_seeds=N_SEEDS,
    n_classes=N_CLASSES,
    run_eegnet=RUN_EEGNET,
):
    loso = LeaveOneGroupOut()
    unique_subjects = np.unique(subjects)

    if X.ndim == 3:
        _, channels, samples = X.shape
    else:
        raise ValueError("Usa datos single-band (3D) para la curva de aprendizaje.")

    model_names = ['CSP', 'MiniRocket'] + (['EEGNet'] if run_eegnet else [])
    results = {name: {n: [] for n in n_trials_list} for name in model_names}

    total_iters = len(n_trials_list) * n_seeds * len(unique_subjects)
    done = 0

    print(f"\n{'='*60}")
    print(f" Learning Curves  |  n_classes={n_classes}  |  seeds={n_seeds}")
    print(f" Modelos: {model_names}")
    print(f" Total iteraciones: {total_iters}")
    print(f"{'='*60}\n")

    for n_trials in n_trials_list:
        for seed in range(n_seeds):
            fold_accs = {name: [] for name in model_names}

            for fold, (tr_idx, te_idx) in enumerate(
                loso.split(X, y, subjects)
            ):
                X_tr_full = X[tr_idx]
                y_tr_full = y[tr_idx]
                X_te      = X[te_idx]
                y_te      = y[te_idx]

                sub_idx = subsample_stratified(
                    X_tr_full, y_tr_full, n_trials, seed=seed
                )
                X_tr = X_tr_full[sub_idx]
                y_tr = y_tr_full[sub_idx]

                subj_label = subjects[te_idx][0]

                try:
                    acc = eval_csp(X_tr, y_tr, X_te, y_te, CSP_CONFIG)
                    fold_accs['CSP'].append(acc)
                except Exception as e:
                    print(f"  [CSP] fold={fold+1} n={n_trials} seed={seed}: {e}")

                try:
                    acc = eval_minirocket(
                        X_tr, y_tr, X_te, y_te, MINIROCKET_CONFIG
                    )
                    fold_accs['MiniRocket'].append(acc)
                except Exception as e:
                    print(f"  [MiniRocket] fold={fold+1} n={n_trials} seed={seed}: {e}")

                if run_eegnet:
                    try:
                        acc = eval_eegnet(
                            X_tr, y_tr, X_te, y_te,
                            channels, samples, n_classes, EEGNET_CONFIG,
                        )
                        fold_accs['EEGNet'].append(acc)
                    except Exception as e:
                        print(f"  [EEGNet] fold={fold+1} n={n_trials} seed={seed}: {e}")

                done += 1

            for name in model_names:
                results[name][n_trials].extend(fold_accs[name])

            csp_mean = np.mean(fold_accs['CSP']) if fold_accs['CSP'] else float('nan')
            mr_mean  = np.mean(fold_accs['MiniRocket']) if fold_accs['MiniRocket'] else float('nan')
            en_str   = (f" | EEGNet={np.mean(fold_accs['EEGNet']):.3f}"
                        if run_eegnet and fold_accs['EEGNet'] else "")
            print(
                f"  n={n_trials:>5} | seed={seed} | "
                f"CSP={csp_mean:.3f} | MiniRocket={mr_mean:.3f}{en_str}"
                f"  [{done}/{total_iters}]"
            )

    return results



def plot_learning_curves(
    results,
    n_trials_list,
    within_subject_n,
    n_classes=2,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'CSP':        '#2ecc71',
        'MiniRocket': '#ffe91f',
        'EEGNet':     '#e74c3c',
    }
    chance = 1.0 / n_classes

    ax.axhline(
        chance, color='gray', linestyle='--', linewidth=1.2,
        alpha=0.7, label=f'Chance level ({chance:.2f})',
        zorder=1,
    )
    ax.axvline(
        within_subject_n, color='steelblue', linestyle=':', linewidth=1.5,
        alpha=0.8, zorder=1,
    )
    ax.text(
        within_subject_n * 1.05, chance + 0.01,
        f'Within-subject\n({within_subject_n} trials)',
        color='steelblue', fontsize=9, va='bottom',
    )

    for model_name, color in colors.items():
        if model_name not in results:
            continue

        xs, means, stds = [], [], []
        for n in n_trials_list:
            accs = results[model_name][n]
            if accs:
                xs.append(n)
                means.append(np.mean(accs))
                stds.append(np.std(accs))

        if not xs:
            continue

        xs    = np.array(xs)
        means = np.array(means)
        stds  = np.array(stds)

        ax.plot(
            xs, means, 'o-',
            color=color, label=model_name,
            linewidth=2.2, markersize=7, zorder=3,
        )
        ax.fill_between(
            xs, means - stds, means + stds,
            alpha=0.15, color=color, zorder=2,
        )

    ax.set_xscale('log')
    ax.set_xticks(n_trials_list)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Number of training trials (log scale)', fontsize=12)
    ax.set_ylabel('Accuracy (mean ± std across folds × seeds)', fontsize=12)
    task = 'Binary (Left vs. Right Hand)' if n_classes == 2 else '4-Class'
    ax.set_title(
        f'Learning Curves — {task} Classification\n'
        f'(LOSO, subsampled training set)',
        fontsize=13,
    )
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(bottom=max(0, chance - 0.05))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nGráfico guardado en: {save_path}")

    plt.show()
    return fig


def print_summary_table(results, n_trials_list):
    model_names = list(results.keys())
    col_w = 12

    header = f"{'n_trials':>10}" + "".join(f"{m:>{col_w}}" for m in model_names)
    print("\n" + "="*len(header))
    print(" Summary — Mean Accuracy (LOSO, subsampled)")
    print("="*len(header))
    print(header)
    print("-"*len(header))

    for n in n_trials_list:
        row = f"{n:>10}"
        for m in model_names:
            accs = results[m].get(n, [])
            val  = f"{np.mean(accs):.4f}" if accs else "  n/a  "
            row += f"{val:>{col_w}}"
        print(row)

    print("="*len(header) + "\n")




if __name__ == "__main__":

    from load_data_BCICIV import load_combined

    for n_classes in [2, 4]:
        X, y, subjects = load_combined(
            data_dir   = 'datasets/BCICIV_2a_gdf',
            label_dir  = 'datasets/BCICIV_2a_gdf/true_labels',
            n_classes  = n_classes,
            use_multiband   = False,
            channels_to_use = None,
        )

        within_subject_n = 144 if n_classes == 2 else 288

        results = compute_learning_curves(
            X, y, subjects,
            n_trials_list = N_TRIALS_LIST,
            n_seeds       = N_SEEDS,
            n_classes     = n_classes,
            run_eegnet    = RUN_EEGNET,
        )

        print_summary_table(results, N_TRIALS_LIST)

        plot_learning_curves(
            results,
            n_trials_list    = N_TRIALS_LIST,
            within_subject_n = within_subject_n,
            n_classes        = n_classes,
            save_path        = f"learning_curve_{n_classes}class.png",
        )
