"""
Main pipeline: Run all steps end-to-end.

Usage: python main.py
"""

import feature_engineering
import train_evaluate
import experiments
import visualize

if __name__ == '__main__':
    print("=" * 60)
    print("LEGO Theme Classification Pipeline")
    print("=" * 60)

    feature_engineering.run()
    train_evaluate.run()
    experiments.run()
    visualize.run()

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("  Dataset:     output/dataset.csv")
    print("  Results:     output/baseline_results.json")
    print("  Experiments: output/experiment_results.json")
    print("  Figures:     figures/")
    print("  Log:         doc/experiment_log.md")
    print("=" * 60)
