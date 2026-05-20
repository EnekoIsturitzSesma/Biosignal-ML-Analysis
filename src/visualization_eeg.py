import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_deep(models, subjects, model_label='Model', save_dir='../figures', CHANCE=0.5):
    SAVE_DIR = save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    names      = models['name']
    accs       = models['accuracies']     
    acc_matrix = np.array(accs)         
    means      = acc_matrix.mean(axis=1)
    stds       = acc_matrix.std(axis=1)
    n_subjects = len(np.unique(subjects))
    subj_labels = [f'S{s}' for s in np.unique(subjects)]


    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'{model_label} — Model Comparison (LOSO, 9 subjects)', fontsize=13)

    ax = axes[0]
    colors = ['tomato' if m == means.max() else 'steelblue' for m in means]
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.8, error_kw=dict(elinewidth=1.2))
    ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
    ax.axhline(CHANCE, color='gray', linestyle='--', label='Chance (0.50)')
    ax.set_ylim(0.3, 1.0); ax.set_ylabel('Mean Accuracy ± std')
    ax.set_title('Mean accuracy per model')
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=0.4)

    ax = axes[1]
    bp = ax.boxplot(acc_matrix.T, labels=names, patch_artist=True,
                    medianprops=dict(color='red', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue'); patch.set_alpha(0.5)
    for i, col in enumerate(acc_matrix):
        ax.scatter([i+1]*n_subjects, col, alpha=0.7, color='steelblue', s=25, zorder=3)
    ax.axhline(CHANCE, color='gray', linestyle='--', label='Chance')
    ax.set_ylabel('Accuracy'); ax.set_title('Per-subject distribution per model')
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/{model_label}_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(acc_matrix, aspect='auto', cmap='RdYlGn', vmin=0.4, vmax=1.0)
    plt.colorbar(im, ax=ax, label='Accuracy')
    ax.set_xticks(range(n_subjects));   ax.set_xticklabels(subj_labels, fontsize=10)
    ax.set_yticks(range(len(names)));   ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(n_subjects):
            ax.text(j, i, f'{acc_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='black' if 0.45 < acc_matrix[i,j] < 0.85 else 'white')
    ax.set_title('Accuracy heatmap — Models × Subjects\n(green = better, red = near chance)')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/{model_label}_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


    feat_configs = ['raw', 'laplacian', 'mu', 'multiband']
    pairs = [
        (f'{f}_22', f'{f}_11', f)
        for f in feat_configs
        if f'{f}_22' in names and f'{f}_11' in names
    ]

    x = np.arange(len(pairs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    m22 = [means[names.index(a)] for a,_,_ in pairs]
    m11 = [means[names.index(b)] for _,b,_ in pairs]
    s22 = [stds [names.index(a)] for a,_,_ in pairs]
    s11 = [stds [names.index(b)] for _,b,_ in pairs]
    labels = [l for _,_,l in pairs]

    b1 = ax.bar(x - w/2, m22, w, yerr=s22, capsize=5, label='22 channels', color='steelblue', alpha=0.8)
    b2 = ax.bar(x + w/2, m11, w, yerr=s11, capsize=5, label='11 channels', color='coral',     alpha=0.8)
    ax.bar_label(b1, fmt='%.3f', fontsize=9); ax.bar_label(b2, fmt='%.3f', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.axhline(CHANCE, color='gray', linestyle='--', label='Chance')
    ax.set_ylim(0.3, 1.0); ax.set_ylabel('Mean Accuracy ± std')
    ax.set_title('22 vs 11 channels — effect per feature configuration')
    ax.legend(); ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/{model_label}_channels.png', dpi=150, bbox_inches='tight')
    plt.show()


    aug_pairs = [
        ('laplacian_22', 'laplacian_22_aug', '22ch + Laplacian'),
        ('laplacian_11', 'laplacian_11_aug', '11ch + Laplacian'),
    ]
    available = [(a, b, lbl) for a, b, lbl in aug_pairs if a in names and b in names]

    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(7*len(available), 5))
        if len(available) == 1: axes = [axes]

        for ax, (no_aug, aug, lbl) in zip(axes, available):
            no_aug_accs = acc_matrix[names.index(no_aug)]
            aug_accs    = acc_matrix[names.index(aug)]
            delta       = aug_accs - no_aug_accs

            xp = np.arange(n_subjects)
            ax.bar(xp - w/2, no_aug_accs, w, label='No aug', color='steelblue', alpha=0.8)
            ax.bar(xp + w/2, aug_accs,    w, label='Aug',    color='coral',     alpha=0.8)

            for j in range(n_subjects):
                color = 'green' if delta[j] > 0 else 'red'
                ax.annotate('', xy=(j + w/2, aug_accs[j] + 0.01),
                            xytext=(j - w/2, no_aug_accs[j] + 0.01),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

            ax.set_xticks(xp); ax.set_xticklabels(subj_labels)
            ax.axhline(CHANCE, color='gray', linestyle='--')
            ax.set_ylim(0.3, 1.0); ax.set_ylabel('Accuracy')
            ax.set_title(f'Augmentation effect — {lbl}\n(→ green = improved, red = degraded)')
            ax.legend(); ax.grid(axis='y', alpha=0.4)

        plt.tight_layout()
        plt.savefig(f'{SAVE_DIR}{model_label}_augmentation.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Best {model_label} model: {names[np.argmax(means)]} — {means.max():.4f}")


def visualize_csp(models, CHANCE=0.5):
    SAVE_DIR = '../figures'
    os.makedirs(SAVE_DIR, exist_ok=True)
    names_csp = models['name']
    scores = np.array(models['score'])     
    acc_csp = np.array(models['subject_scores'])

    configs = list(dict.fromkeys(n.rsplit('_', 1)[0] for n in names_csp))  
    lda_scores = [scores[names_csp.index(f'{c}_LDA')] for c in configs if f'{c}_LDA' in names_csp]
    svm_scores = [scores[names_csp.index(f'{c}_SVM')] for c in configs if f'{c}_SVM' in names_csp]

    x   = np.arange(len(configs))
    w   = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle('CSP Classifiers — LOSO comparison', fontsize=13)

    ax = axes[0]
    b1 = ax.bar(x - w/2, lda_scores, w, label='LDA', color='steelblue', alpha=0.8)
    b2 = ax.bar(x + w/2, svm_scores, w, label='SVM', color='coral',     alpha=0.8)
    ax.bar_label(b1, fmt='%.3f', fontsize=8); ax.bar_label(b2, fmt='%.3f', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.axhline(CHANCE, color='gray', linestyle='--', label='Chance')
    ax.set_ylim(0.3, 0.6); ax.set_ylabel('Mean CV Accuracy')
    ax.set_title('LDA vs SVM per feature configuration')
    ax.legend(); ax.grid(axis='y', alpha=0.4)

    configs_22 = [n for n in names_csp if '22' in n]
    configs_11 = [n for n in names_csp if '11' in n]
    ax = axes[1]
    bp_data = [scores[[names_csp.index(n) for n in configs_22]],
            scores[[names_csp.index(n) for n in configs_11]]]
    bp = ax.boxplot(bp_data, labels=['22 channels', '11 channels'], patch_artist=True,
                    medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor('steelblue'); bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('coral');     bp['boxes'][1].set_alpha(0.5)
    for i, d in enumerate(bp_data):
        ax.scatter([i+1]*len(d), d, alpha=0.8, zorder=3, s=30,
                color=['steelblue', 'coral'][i])
    ax.axhline(CHANCE, color='gray', linestyle='--', label='Chance')
    ax.set_ylabel('CV Accuracy'); ax.set_title('22 vs 11 channels — all CSP configs')
    ax.set_ylim(0.48, 0.54); ax.legend(); ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/csp_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nBest CSP model: {names_csp[np.argmax(scores)]} — {scores.max():.4f}")
