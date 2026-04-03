"""
Experiments: Investigate how dataset characteristics affect model performance.

Experiment 1: Effect of training data amount
Experiment 2: Class balance strategies (original / balanced / SMOTE)
Experiment 3: PCA dimensionality reduction
Experiment 4: Feature ablation (color-only / category-only / metadata-only / all)
Experiment 5: Hyperparameter sensitivity (Random Forest)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone
from sklearn.utils import resample

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
DOC_DIR = os.path.join(os.path.dirname(__file__), 'doc')


def load_dataset():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'dataset.csv'))
    feature_cols = [c for c in df.columns if c not in ['set_num', 'theme']]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df['theme'])
    return X, y, feature_cols, le, df


def get_models_fast():
    """Models for experiments that run many iterations (exp1)."""
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=None, class_weight='balanced',
            random_state=42, n_jobs=-1),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=200, max_depth=8, learning_rate=0.1, random_state=42),
    }


def get_models():
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight='balanced',
            random_state=42, n_jobs=-1),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=300, max_depth=8, learning_rate=0.1, random_state=42),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation='relu',
            max_iter=300, early_stopping=True, validation_fraction=0.15,
            random_state=42),
    }


def cv_evaluate(X, y, model_template, n_splits=5):
    """Run stratified CV and return mean accuracy and F1."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = clone(model_template)
        model.fit(X_train, y[train_idx])
        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred, average='macro', zero_division=0))
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


# ============================================================
# Experiment 1: Training data amount
# ============================================================
def exp1_data_amount(X, y):
    print("\n=== Experiment 1: Training Data Amount ===")
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    models = get_models_fast()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for model_name, model_template in models.items():
        for frac in fractions:
            accs = []
            for train_idx, test_idx in skf.split(X, y):
                X_train_full, y_train_full = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                # Subsample training data
                if frac < 1.0:
                    n_sub = max(1, int(len(X_train_full) * frac))
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(X_train_full), n_sub, replace=False)
                    X_train_sub, y_train_sub = X_train_full[idx], y_train_full[idx]
                else:
                    X_train_sub, y_train_sub = X_train_full, y_train_full

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train_sub)
                X_te = scaler.transform(X_test)
                model = clone(model_template)
                model.fit(X_tr, y_train_sub)
                accs.append(accuracy_score(y_test, model.predict(X_te)))

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            results.append({'model': model_name, 'fraction': frac,
                            'accuracy': mean_acc, 'accuracy_std': std_acc})
            print(f"  {model_name}, frac={frac:.1f}: Acc={mean_acc:.4f}±{std_acc:.4f}")

    return results


# ============================================================
# Experiment 2: Class balance strategies
# ============================================================
def exp2_class_balance(X, y):
    print("\n=== Experiment 2: Class Balance Strategies ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    strategies = {
        'Original': {},
        'Balanced (class_weight)': {'class_weight': 'balanced'},
    }

    # RF and HGB with different class_weight settings
    for strategy_name, extra_params in strategies.items():
        for model_name_base, model_class, default_params in [
            ('Random Forest', RandomForestClassifier,
             dict(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)),
            ('Hist Gradient Boosting', HistGradientBoostingClassifier,
             dict(max_iter=300, max_depth=8, learning_rate=0.1, random_state=42)),
        ]:
            params = {**default_params}
            if 'class_weight' in extra_params and model_class == RandomForestClassifier:
                params['class_weight'] = extra_params['class_weight']
            elif 'class_weight' in extra_params and model_class == HistGradientBoostingClassifier:
                params['class_weight'] = extra_params['class_weight']

            model_template = model_class(**params)
            acc_m, acc_s, f1_m, f1_s = cv_evaluate(X, y, model_template)
            results.append({'model': model_name_base, 'strategy': strategy_name,
                            'accuracy': acc_m, 'accuracy_std': acc_s,
                            'f1': f1_m, 'f1_std': f1_s})
            print(f"  {model_name_base}, {strategy_name}: Acc={acc_m:.4f}, F1={f1_m:.4f}")

    # SMOTE
    print("  Running SMOTE experiments...")
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        for model_name_base, model_template in [
            ('Random Forest', RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)),
            ('Hist Gradient Boosting', HistGradientBoostingClassifier(
                max_iter=300, max_depth=8, learning_rate=0.1, random_state=42)),
        ]:
            accs, f1s = [], []
            for train_idx, test_idx in skf.split(X, y):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                X_res, y_res = smote.fit_resample(X_train, y[train_idx])
                model = clone(model_template)
                model.fit(X_res, y_res)
                y_pred = model.predict(X_test)
                accs.append(accuracy_score(y[test_idx], y_pred))
                f1s.append(f1_score(y[test_idx], y_pred, average='macro', zero_division=0))
            results.append({'model': model_name_base, 'strategy': 'SMOTE',
                            'accuracy': np.mean(accs), 'accuracy_std': np.std(accs),
                            'f1': np.mean(f1s), 'f1_std': np.std(f1s)})
            print(f"  {model_name_base}, SMOTE: Acc={np.mean(accs):.4f}, F1={np.mean(f1s):.4f}")
    except ImportError:
        print("  imbalanced-learn not installed, skipping SMOTE")

    return results


# ============================================================
# Experiment 3: PCA dimensionality reduction
# ============================================================
def exp3_pca(X, y):
    print("\n=== Experiment 3: PCA Dimensionality Reduction ===")
    n_components_list = [10, 30, 50, 100, 150]
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight='balanced',
        random_state=42, n_jobs=-1)

    results = []

    # Full PCA for explained variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    explained_var = pca_full.explained_variance_ratio_.cumsum().tolist()

    # Baseline: no PCA
    acc_m, acc_s, f1_m, f1_s = cv_evaluate(X, y, rf)
    results.append({'n_components': X.shape[1], 'accuracy': acc_m,
                    'accuracy_std': acc_s, 'f1': f1_m, 'label': 'Full'})
    print(f"  Full ({X.shape[1]} dims): Acc={acc_m:.4f}")

    # PCA variants
    for n_comp in n_components_list:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, f1s = [], []
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            pca = PCA(n_components=n_comp, random_state=42)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            model = clone(rf)
            model.fit(X_train_pca, y[train_idx])
            y_pred = model.predict(X_test_pca)
            accs.append(accuracy_score(y[test_idx], y_pred))
            f1s.append(f1_score(y[test_idx], y_pred, average='macro', zero_division=0))

        results.append({'n_components': n_comp, 'accuracy': np.mean(accs),
                        'accuracy_std': np.std(accs), 'f1': np.mean(f1s),
                        'label': str(n_comp)})
        print(f"  PCA({n_comp}): Acc={np.mean(accs):.4f}")

    return results, explained_var


# ============================================================
# Experiment 4: Feature ablation
# ============================================================
def exp4_feature_ablation(X, y, feature_cols):
    print("\n=== Experiment 4: Feature Ablation ===")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight='balanced',
        random_state=42, n_jobs=-1)

    # Define feature groups
    color_idx = [i for i, c in enumerate(feature_cols) if c.startswith('color_')]
    cat_idx = [i for i, c in enumerate(feature_cols) if c.startswith('cat_')]
    color_stat_idx = [i for i, c in enumerate(feature_cols)
                      if c in ['avg_r', 'avg_g', 'avg_b', 'std_r', 'std_g', 'std_b', 'prop_transparent']]
    meta_idx = [i for i, c in enumerate(feature_cols)
                if c in ['num_parts', 'year', 'num_unique_parts', 'num_unique_colors',
                         'num_unique_categories', 'num_minifigs', 'num_unique_minifigs']
                or c.startswith('mat_')]

    groups = {
        'Color features': color_idx + color_stat_idx,
        'Part category features': cat_idx,
        'Metadata features': meta_idx,
        'All features': list(range(len(feature_cols))),
    }

    results = []
    for group_name, idx in groups.items():
        X_sub = X[:, idx]
        acc_m, acc_s, f1_m, f1_s = cv_evaluate(X_sub, y, rf)
        results.append({'feature_group': group_name, 'n_features': len(idx),
                        'accuracy': acc_m, 'accuracy_std': acc_s,
                        'f1': f1_m, 'f1_std': f1_s})
        print(f"  {group_name} ({len(idx)} dims): Acc={acc_m:.4f}, F1={f1_m:.4f}")

    return results


# ============================================================
# Experiment 5: Hyperparameter sensitivity
# ============================================================
def exp5_hyperparameters(X, y):
    print("\n=== Experiment 5: Hyperparameter Sensitivity ===")
    results_estimators = []
    results_depth = []

    # Vary n_estimators
    for n_est in [10, 50, 100, 200, 500]:
        rf = RandomForestClassifier(
            n_estimators=n_est, max_depth=None, class_weight='balanced',
            random_state=42, n_jobs=-1)
        acc_m, acc_s, f1_m, f1_s = cv_evaluate(X, y, rf)
        results_estimators.append({'n_estimators': n_est, 'accuracy': acc_m,
                                   'accuracy_std': acc_s, 'f1': f1_m})
        print(f"  n_estimators={n_est}: Acc={acc_m:.4f}")

    # Vary max_depth
    for depth in [5, 10, 20, 50, None]:
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=depth, class_weight='balanced',
            random_state=42, n_jobs=-1)
        acc_m, acc_s, f1_m, f1_s = cv_evaluate(X, y, rf)
        depth_label = str(depth) if depth else 'None'
        results_depth.append({'max_depth': depth_label, 'accuracy': acc_m,
                              'accuracy_std': acc_s, 'f1': f1_m})
        print(f"  max_depth={depth_label}: Acc={acc_m:.4f}")

    return results_estimators, results_depth


def run():
    X, y, feature_cols, le, df = load_dataset()

    all_results = {}
    all_results['exp1'] = exp1_data_amount(X, y)
    all_results['exp2'] = exp2_class_balance(X, y)
    all_results['exp3_results'], all_results['exp3_explained_var'] = exp3_pca(X, y)
    all_results['exp4'] = exp4_feature_ablation(X, y, feature_cols)
    all_results['exp5_estimators'], all_results['exp5_depth'] = exp5_hyperparameters(X, y)

    # Save all experiment results
    results_path = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Write experiment log
    os.makedirs(DOC_DIR, exist_ok=True)
    log_path = os.path.join(DOC_DIR, 'experiment_log.md')
    with open(log_path, 'w') as f:
        f.write("# Experiment Log\n\n")

        f.write("## Experiment 1: Training Data Amount\n\n")
        f.write("| Model | 20% | 40% | 60% | 80% | 100% |\n")
        f.write("|-------|-----|-----|-----|-----|------|\n")
        exp1_df = pd.DataFrame(all_results['exp1'])
        for model in exp1_df['model'].unique():
            row = exp1_df[exp1_df['model'] == model]
            vals = [f"{row[row['fraction']==fr]['accuracy'].values[0]:.4f}" for fr in [0.2,0.4,0.6,0.8,1.0]]
            f.write(f"| {model} | {' | '.join(vals)} |\n")

        f.write("\n## Experiment 2: Class Balance Strategies\n\n")
        f.write("| Model | Strategy | Accuracy | F1 (Macro) |\n")
        f.write("|-------|----------|----------|------------|\n")
        for r in all_results['exp2']:
            f.write(f"| {r['model']} | {r['strategy']} | {r['accuracy']:.4f} | {r['f1']:.4f} |\n")

        f.write("\n## Experiment 3: PCA Dimensionality Reduction\n\n")
        f.write("| Components | Accuracy |\n")
        f.write("|------------|----------|\n")
        for r in all_results['exp3_results']:
            f.write(f"| {r['label']} | {r['accuracy']:.4f} |\n")

        f.write("\n## Experiment 4: Feature Ablation\n\n")
        f.write("| Feature Group | # Features | Accuracy | F1 (Macro) |\n")
        f.write("|---------------|------------|----------|------------|\n")
        for r in all_results['exp4']:
            f.write(f"| {r['feature_group']} | {r['n_features']} | {r['accuracy']:.4f} | {r['f1']:.4f} |\n")

        f.write("\n## Experiment 5: Hyperparameter Sensitivity\n\n")
        f.write("### n_estimators\n\n")
        f.write("| n_estimators | Accuracy |\n")
        f.write("|--------------|----------|\n")
        for r in all_results['exp5_estimators']:
            f.write(f"| {r['n_estimators']} | {r['accuracy']:.4f} |\n")

        f.write("\n### max_depth\n\n")
        f.write("| max_depth | Accuracy |\n")
        f.write("|-----------|----------|\n")
        for r in all_results['exp5_depth']:
            f.write(f"| {r['max_depth']} | {r['accuracy']:.4f} |\n")

    print(f"\nExperiment results saved to {results_path}")
    print(f"Experiment log saved to {log_path}")
    return all_results


if __name__ == '__main__':
    run()
