"""
Visualization: Generate all figures for the report.

Reads results from output/ and produces PNG figures in figures/.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE

# Global style
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def load_results():
    with open(os.path.join(OUTPUT_DIR, 'baseline_results.json')) as f:
        baseline = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'experiment_results.json')) as f:
        experiments = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'class_names.json')) as f:
        class_names = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'feature_names.json')) as f:
        feature_names = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'feature_importance.json')) as f:
        importance = json.load(f)
    return baseline, experiments, class_names, feature_names, importance


def fig1_class_distribution():
    """Bar chart of samples per theme."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'dataset.csv'))
    counts = df['theme'].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette('Set2', len(counts))
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel('Number of Sets')
    ax.set_title('Dataset: Samples per LEGO Theme')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=10)
    plt.savefig(os.path.join(FIG_DIR, 'class_distribution.png'))
    plt.close()
    print("  [1/11] class_distribution.png")


def fig2_tsne():
    """t-SNE visualization of feature space."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'dataset.csv'))
    feature_cols = [c for c in df.columns if c not in ['set_num', 'theme']]
    X = df[feature_cols].values
    y = df['theme'].values

    # Subsample for speed
    rng = np.random.RandomState(42)
    n = min(3000, len(X))
    idx = rng.choice(len(X), n, replace=False)
    X_sub, y_sub = X[idx], y[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_scaled)

    themes = sorted(set(y_sub))
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette('tab10', len(themes))
    for i, theme in enumerate(themes):
        mask = y_sub == theme
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[palette[i]], label=theme,
                   alpha=0.6, s=15, edgecolors='none')
    ax.legend(fontsize=9, markerscale=2, loc='best')
    ax.set_title('t-SNE Visualization of LEGO Set Features')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.savefig(os.path.join(FIG_DIR, 'tsne_visualization.png'))
    plt.close()
    print("  [2/11] tsne_visualization.png")


def fig3_confusion_matrices(baseline, class_names):
    """Confusion matrix heatmaps for each model."""
    for i, (model_name, res) in enumerate(baseline.items()):
        cm = np.array(res['confusion_matrix'])
        # Normalize by row (true labels)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    vmin=0, vmax=1, annot_kws={'size': 9})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix: {model_name}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        fname = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(FIG_DIR, fname))
        plt.close()
        print(f"  [{3+i}/11] {fname}")


def fig4_learning_curves(experiments):
    """Accuracy vs training data fraction for each model."""
    exp1 = pd.DataFrame(experiments['exp1'])

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^']
    for i, model in enumerate(exp1['model'].unique()):
        data = exp1[exp1['model'] == model]
        ax.errorbar(data['fraction'], data['accuracy'], yerr=data['accuracy_std'],
                    marker=markers[i], label=model, capsize=3, linewidth=2)
    ax.set_xlabel('Training Data Fraction')
    ax.set_ylabel('Accuracy')
    ax.set_title('Effect of Training Data Amount')
    ax.legend()
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, 'learning_curves.png'))
    plt.close()
    print("  [6/11] learning_curves.png")


def fig5_pca_explained_variance(experiments):
    """Cumulative explained variance vs number of components."""
    explained = experiments['exp3_explained_var']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(explained)+1), explained, linewidth=2, color='steelblue')
    # Mark key thresholds
    for thresh in [0.90, 0.95, 0.99]:
        n = next((i+1 for i, v in enumerate(explained) if v >= thresh), len(explained))
        ax.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=n, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'{thresh*100:.0f}% at n={n}', xy=(n, thresh),
                    fontsize=9, ha='left', va='bottom')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA: Explained Variance')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, 'pca_explained_variance.png'))
    plt.close()
    print("  [7/11] pca_explained_variance.png")


def fig6_pca_accuracy(experiments):
    """Accuracy vs number of PCA components."""
    pca_results = experiments['exp3_results']

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r['label'] for r in pca_results]
    accs = [r['accuracy'] for r in pca_results]
    x_pos = range(len(labels))
    bars = ax.bar(x_pos, accs, color='steelblue', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Number of PCA Components')
    ax.set_ylabel('Accuracy')
    ax.set_title('PCA: Effect on Classification Accuracy (Random Forest)')
    ax.set_ylim(min(accs) - 0.02, max(accs) + 0.01)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(FIG_DIR, 'pca_accuracy.png'))
    plt.close()
    print("  [8/11] pca_accuracy.png")


def fig7_class_balance(experiments):
    """Grouped bar chart: F1 per model per balancing strategy."""
    exp2 = pd.DataFrame(experiments['exp2'])

    fig, ax = plt.subplots(figsize=(9, 5))
    models = exp2['model'].unique()
    strategies = exp2['strategy'].unique()
    x = np.arange(len(models))
    width = 0.25

    for i, strategy in enumerate(strategies):
        data = exp2[exp2['strategy'] == strategy]
        f1_vals = [data[data['model'] == m]['f1'].values[0] for m in models]
        bars = ax.bar(x + i * width, f1_vals, width, label=strategy, alpha=0.85)

    ax.set_xlabel('Model')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Effect of Class Balance Strategy')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(FIG_DIR, 'class_balance_comparison.png'))
    plt.close()
    print("  [9/11] class_balance_comparison.png")


def fig8_feature_importance(importance, feature_names):
    """Top 30 features by Random Forest importance."""
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:30]
    names = [x[0] for x in sorted_imp][::-1]
    values = [x[1] for x in sorted_imp][::-1]

    # Map feature names to readable labels
    def readable_name(name):
        if name.startswith('cat_'):
            # Load part categories
            try:
                cats = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'part_categories.csv'))
                cat_id = int(name.replace('cat_', ''))
                match = cats[cats['id'] == cat_id]
                if not match.empty:
                    return f"Cat: {match.iloc[0]['name']}"
            except:
                pass
        elif name.startswith('color_'):
            try:
                colors = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'colors.csv'))
                color_id = int(name.replace('color_', ''))
                match = colors[colors['id'] == color_id]
                if not match.empty:
                    return f"Color: {match.iloc[0]['name']}"
            except:
                pass
        return name

    readable = [readable_name(n) for n in names]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(range(len(readable)), values, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(readable)))
    ax.set_yticklabels(readable, fontsize=9)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 30 Features (Random Forest)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.savefig(os.path.join(FIG_DIR, 'feature_importance.png'))
    plt.close()
    print("  [10/11] feature_importance.png")


def fig9_feature_ablation(experiments):
    """Bar chart: accuracy per feature subset."""
    exp4 = experiments['exp4']

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [r['feature_group'] for r in exp4]
    accs = [r['accuracy'] for r in exp4]
    f1s = [r['f1'] for r in exp4]
    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', alpha=0.85)
    bars2 = ax.bar(x + width/2, f1s, width, label='Macro F1', alpha=0.85)

    ax.set_ylabel('Score')
    ax.set_title('Feature Ablation Study')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha='right', fontsize=10)
    ax.legend()
    ax.set_ylim(min(min(accs), min(f1s)) - 0.05, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.savefig(os.path.join(FIG_DIR, 'feature_ablation.png'))
    plt.close()
    print("  [11/11] feature_ablation.png")


def fig10_hyperparams(experiments):
    """Hyperparameter sensitivity plots."""
    # n_estimators
    est_data = experiments['exp5_estimators']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = [r['n_estimators'] for r in est_data]
    y = [r['accuracy'] for r in est_data]
    ax.plot(x, y, 'o-', linewidth=2, color='steelblue')
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Accuracy')
    ax.set_title('RF: Effect of n_estimators')
    ax.grid(True, alpha=0.3)

    # max_depth
    depth_data = experiments['exp5_depth']
    ax = axes[1]
    labels = [r['max_depth'] for r in depth_data]
    y = [r['accuracy'] for r in depth_data]
    ax.plot(range(len(labels)), y, 'o-', linewidth=2, color='coral')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('max_depth')
    ax.set_ylabel('Accuracy')
    ax.set_title('RF: Effect of max_depth')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'hyperparameter_sensitivity.png'))
    plt.close()
    print("  [bonus] hyperparameter_sensitivity.png")


def run():
    print("=== Generating Figures ===")
    os.makedirs(FIG_DIR, exist_ok=True)

    baseline, experiments, class_names, feature_names, importance = load_results()

    fig1_class_distribution()
    fig2_tsne()
    fig3_confusion_matrices(baseline, class_names)
    fig4_learning_curves(experiments)
    fig5_pca_explained_variance(experiments)
    fig6_pca_accuracy(experiments)
    fig7_class_balance(experiments)
    fig8_feature_importance(importance, feature_names)
    fig9_feature_ablation(experiments)
    fig10_hyperparams(experiments)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    run()
