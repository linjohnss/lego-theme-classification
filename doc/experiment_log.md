# Experiment Log

## Experiment 1: Training Data Amount

| Model | 20% | 40% | 60% | 80% | 100% |
|-------|-----|-----|-----|-----|------|
| Random Forest | 0.9002 | 0.9197 | 0.9281 | 0.9357 | 0.9384 |
| Hist Gradient Boosting | 0.9119 | 0.9319 | 0.9428 | 0.9494 | 0.9549 |

## Experiment 2: Class Balance Strategies

| Model | Strategy | Accuracy | F1 (Macro) |
|-------|----------|----------|------------|
| Random Forest | Original | 0.9360 | 0.9110 |
| Hist Gradient Boosting | Original | 0.9550 | 0.9392 |
| Random Forest | Balanced (class_weight) | 0.9383 | 0.9145 |
| Hist Gradient Boosting | Balanced (class_weight) | 0.9543 | 0.9390 |
| Random Forest | SMOTE | 0.9397 | 0.9207 |
| Hist Gradient Boosting | SMOTE | 0.9491 | 0.9328 |

## Experiment 3: PCA Dimensionality Reduction

| Components | Accuracy |
|------------|----------|
| Full | 0.9383 |
| 10 | 0.8492 |
| 30 | 0.8839 |
| 50 | 0.8946 |
| 100 | 0.8927 |
| 150 | 0.8932 |

## Experiment 4: Feature Ablation

| Feature Group | # Features | Accuracy | F1 (Macro) |
|---------------|------------|----------|------------|
| Color features | 173 | 0.8213 | 0.7788 |
| Part category features | 70 | 0.8801 | 0.8376 |
| Metadata features | 13 | 0.8077 | 0.7481 |
| All features | 256 | 0.9383 | 0.9145 |

## Experiment 5: Hyperparameter Sensitivity

### n_estimators

| n_estimators | Accuracy |
|--------------|----------|
| 10 | 0.9067 |
| 50 | 0.9317 |
| 100 | 0.9358 |
| 200 | 0.9384 |
| 500 | 0.9376 |

### max_depth

| max_depth | Accuracy |
|-----------|----------|
| 5 | 0.8513 |
| 10 | 0.9153 |
| 20 | 0.9383 |
| 50 | 0.9383 |
| None | 0.9383 |
