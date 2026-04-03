"""
Model Training and Evaluation: Baseline performance with 5-fold Stratified CV.

Models: Random Forest, Histogram Gradient Boosting, MLP (neural network).
Metrics: Accuracy, Macro Precision/Recall/F1, Confusion Matrix.
Results saved to output/ as JSON for use by visualize.py.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


def get_models():
    """Return dict of model name -> model instance."""
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=300, max_depth=8, learning_rate=0.1, random_state=42
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation='relu',
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            random_state=42
        ),
    }


def load_dataset():
    """Load dataset and return X, y, feature_names, label_encoder."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'dataset.csv'))
    feature_cols = [c for c in df.columns if c not in ['set_num', 'theme']]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df['theme'])
    return X, y, feature_cols, le


def run():
    print("=== Model Training & Evaluation ===")
    X, y, feature_names, le = load_dataset()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes")
    print(f"Classes: {list(le.classes_)}\n")

    models = get_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_results = {}

    for model_name, model_template in models.items():
        print(f"--- {model_name} ---")
        fold_metrics = []
        fold_cms = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Clone model (fresh instance each fold)
            from sklearn.base import clone
            model = clone(model_template)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))

            fold_metrics.append({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
            fold_cms.append(cm)
            print(f"  Fold {fold_idx+1}: Acc={acc:.4f}, F1={f1:.4f}")

        # Aggregate
        metrics_df = pd.DataFrame(fold_metrics)
        mean_metrics = metrics_df.mean().to_dict()
        std_metrics = metrics_df.std().to_dict()
        avg_cm = np.mean(fold_cms, axis=0)

        all_results[model_name] = {
            'mean': mean_metrics,
            'std': std_metrics,
            'confusion_matrix': avg_cm.tolist(),
            'fold_metrics': fold_metrics,
        }

        print(f"  Mean: Acc={mean_metrics['accuracy']:.4f}±{std_metrics['accuracy']:.4f}, "
              f"F1={mean_metrics['f1']:.4f}±{std_metrics['f1']:.4f}\n")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save class names
    classes_path = os.path.join(OUTPUT_DIR, 'class_names.json')
    with open(classes_path, 'w') as f:
        json.dump(list(le.classes_), f)

    # Save feature names
    features_path = os.path.join(OUTPUT_DIR, 'feature_names.json')
    with open(features_path, 'w') as f:
        json.dump(feature_names, f)

    # Train final RF on full data for feature importance
    print("Training final Random Forest for feature importance...")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    rf_final = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_final.fit(X_s, y)
    importance = rf_final.feature_importances_
    importance_path = os.path.join(OUTPUT_DIR, 'feature_importance.json')
    with open(importance_path, 'w') as f:
        json.dump(dict(zip(feature_names, importance.tolist())), f)

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Model':<25} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("-" * 73)
    for name, res in all_results.items():
        m, s = res['mean'], res['std']
        print(f"{name:<25} {m['accuracy']:.4f}±{s['accuracy']:.4f} "
              f"{m['precision']:.4f}±{s['precision']:.4f} "
              f"{m['recall']:.4f}±{s['recall']:.4f} "
              f"{m['f1']:.4f}±{s['f1']:.4f}")

    print(f"\nResults saved to {results_path}")
    return all_results


if __name__ == '__main__':
    run()
