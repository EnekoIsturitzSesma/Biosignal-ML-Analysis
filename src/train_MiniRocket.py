import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sktime.transformations.panel.rocket import MiniRocketMultivariate
import mlflow


def _make_transform(cfg):
    return MiniRocketMultivariate(
        num_kernels=cfg.get("num_kernels", 10_000),
        random_state=cfg.get("random_state", 42),
    )


def _make_classifier(cfg):
    alphas = cfg.get("alphas", np.logspace(-3, 3, 10))
    return RidgeClassifierCV(alphas=alphas, class_weight="balanced")


def _fit_predict(X_train, y_train, X_test, cfg):

    transform = _make_transform(cfg)
    transform.fit(X_train)
    X_tr = transform.transform(X_train)
    X_te = transform.transform(X_test)

    scaler = StandardScaler(with_mean=True)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = _make_classifier(cfg)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)

    return y_pred


def run_loso(X, y, subjects, cfg, num_classes=4, run_name="MiniRocket_loso"):
    unique_subjects = np.unique(subjects)
    subject_accs = {}

    mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        "method":        "MiniRocket+Ridge",
        "num_kernels":   cfg.get("num_kernels", 10_000),
        "config":        cfg.get("name", ""),
        "num_classes":   num_classes,
    })


    for subj in unique_subjects:
        test_mask  = subjects == subj
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        y_pred = _fit_predict(X_train, y_train, X_test, cfg)

        acc = np.mean(y_pred == y_test)
        subject_accs[f"s{subj}"] = float(acc)
        print(f"  subject {subj:>2} → acc={acc:.4f}")

    accs = np.array(list(subject_accs.values()))
    mean_acc = float(accs.mean())
    std_acc  = float(accs.std())

    print(f"\n[{run_name}]  mean={mean_acc:.4f} ± {std_acc:.4f}")

    mlflow.log_metrics({"mean_acc": mean_acc, "std_acc": std_acc})
    for subj_key, acc in subject_accs.items():
        mlflow.log_metric(f"acc_{subj_key}", acc)
    mlflow.end_run()

    return {"mean": mean_acc, "std": std_acc, "per_subject": subject_accs}


def run_within_subject_T_to_E(X_T, y_T, subjects_T, X_E, y_E, subjects_E, cfg, num_classes=4, run_name="MiniRocket_within"):
    unique_subjects = np.unique(subjects_T)
    subject_accs = {}

    mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        "method":      "MiniRocket+Ridge",
        "num_kernels": cfg.get("num_kernels", 10_000),
        "config":      cfg.get("name", ""),
        "protocol":    "within_T_to_E",
    })


    for subj in unique_subjects:
        mask_T = subjects_T == subj
        mask_E = subjects_E == subj

        if not mask_E.any():
            print(f"  subject {subj}: no E data, skipping.")
            continue

        X_train, y_train = X_T[mask_T], y_T[mask_T]
        X_test,  y_test  = X_E[mask_E], y_E[mask_E]

        y_pred = _fit_predict(X_train, y_train, X_test, cfg)

        acc = np.mean(y_pred == y_test)
        subject_accs[f"s{subj}"] = float(acc)
        print(f"  subject {subj:>2} → acc={acc:.4f}")

    accs = np.array(list(subject_accs.values()))
    mean_acc = float(accs.mean())
    std_acc  = float(accs.std())

    print(f"\n[{run_name}]  mean={mean_acc:.4f} ± {std_acc:.4f}  ")

    mlflow.log_metrics({"mean_acc": mean_acc, "std_acc": std_acc})
    for subj_key, acc in subject_accs.items():
        mlflow.log_metric(f"acc_{subj_key}", acc)
    mlflow.end_run()

    return {"mean": mean_acc, "std": std_acc, "per_subject": subject_accs}
