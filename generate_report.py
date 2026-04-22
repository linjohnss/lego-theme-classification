"""
Generate compact PDF report (max 10 pages body + appendix).
Student: 312554027 Chin-Yang Lin
"""

import os
import json
import numpy as np
from fpdf import FPDF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

with open(os.path.join(OUTPUT_DIR, 'baseline_results.json')) as f:
    baseline = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'experiment_results.json')) as f:
    experiments = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'class_names.json')) as f:
    class_names = json.load(f)


class Report(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.add_font('NS', '', '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf')
        self.add_font('NS', 'B', '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf')
        self.add_font('NS', 'I', '/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf')
        self.add_font('NS', 'BI', '/usr/share/fonts/truetype/noto/NotoSans-BoldItalic.ttf')
        self.add_font('NM', '', '/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf')
        self.set_auto_page_break(auto=True, margin=15)
        self.is_appendix = False

    def header(self):
        if self.page_no() > 1:
            self.set_font('NS', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 6, 'AI Capstone Project #1 - 312554027', align='L')
            self.cell(0, 6, f'Page {self.page_no()}', align='R', new_x='LMARGIN', new_y='NEXT')
            self.set_text_color(0, 0, 0)

    def s1(self, t):
        self.set_font('NS', 'B', 14)
        self.ln(3)
        self.cell(0, 8, t, new_x='LMARGIN', new_y='NEXT')
        self.ln(1)

    def s2(self, t):
        self.set_font('NS', 'B', 12)
        self.ln(2)
        self.cell(0, 7, t, new_x='LMARGIN', new_y='NEXT')
        self.ln(1)

    def s3(self, t):
        self.set_font('NS', 'B', 11)
        self.ln(1)
        self.cell(0, 6, t, new_x='LMARGIN', new_y='NEXT')

    def p(self, text):
        self.set_font('NS', '', 12)
        self.multi_cell(0, 5.8, text)
        self.ln(1)

    def fig(self, fn, cap, w=130):
        path = os.path.join(FIG_DIR, fn)
        if not os.path.exists(path):
            return
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        self.ln(1)
        self.set_font('NS', 'I', 9)
        self.multi_cell(0, 4.5, cap, align='C')
        self.ln(2)

    def tbl(self, headers, rows, cw=None, fs=9):
        self.set_font('NS', 'B', fs)
        if cw is None:
            cw = [170 / len(headers)] * len(headers)
        # Center table horizontally
        table_w = sum(cw)
        left_margin = (210 - table_w) / 2  # A4 width = 210mm

        self.set_fill_color(70, 130, 180)
        self.set_text_color(255, 255, 255)
        self.set_x(left_margin)
        for i, h in enumerate(headers):
            self.cell(cw[i], 6, h, border=1, align='C', fill=True)
        self.ln()
        self.set_font('NS', '', fs)
        self.set_text_color(0, 0, 0)
        for ri, row in enumerate(rows):
            fc = (240, 248, 255) if ri % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*fc)
            self.set_x(left_margin)
            for i, c in enumerate(row):
                self.cell(cw[i], 5.5, str(c), border=1, align='C', fill=True)
            self.ln()
        self.ln(1)

    def caption(self, text):
        self.set_font('NS', 'I', 9)
        self.cell(0, 4.5, text, align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)


def build_report():
    pdf = Report()

    # ============ PAGE 1: Title + Introduction + Dataset Start ============
    pdf.add_page()
    pdf.set_font('NS', 'B', 18)
    pdf.cell(0, 10, 'Predicting LEGO Set Themes from Part Composition', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)
    pdf.set_font('NS', '', 11)
    pdf.cell(0, 6, 'AI Capstone Project #1  |  NYCU Spring 2026  |  312554027 Chin-Yang Lin', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('NS', '', 10)
    pdf.cell(0, 6, 'Dataset: https://github.com/linjohnss/lego-theme-classification/blob/main/output/dataset.csv', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, 'Code: https://github.com/linjohnss/lego-theme-classification', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)

    # --- 1. Introduction ---
    pdf.s1('1. Introduction')
    pdf.p(
        'In machine learning, dataset quality often matters more than algorithm choice. This project '
        'explores dataset construction through a creative classification task: predicting which LEGO '
        'product theme a set belongs to based solely on its part composition. LEGO themes have '
        'distinctive characteristics. For example, Star Wars uses grey palettes with many minifigures, '
        'Technic uses beams and gears instead of bricks, and Duplo uses entirely different large-format '
        'parts. We hypothesize these differences enable accurate classification from parts lists alone.'
    )
    pdf.p(
        'Our research questions: (1) Can ML models reliably predict LEGO themes from part-level '
        'features? (2) Which feature groups contribute most? (3) How do class balance, data volume, '
        'and dimensionality reduction affect performance?'
    )

    # --- 2. Dataset ---
    pdf.s1('2. Dataset')
    pdf.s2('2.1 Source and Construction')
    pdf.p(
        'We constructed our dataset from the Rebrickable open database (rebrickable.com/downloads/), '
        'which provides comprehensive LEGO data as CSV files. We downloaded 9 tables (~1.6M records) '
        'using Python\'s requests library. While the raw data is public, our dataset is self-constructed: '
        'we designed the feature pipeline, selected themes, and computed all features from scratch. '
        'This is similar to using government weather APIs as a source for a custom prediction dataset.'
    )
    pdf.s2('2.2 Data Cleanup and Processing')
    pdf.p(
        'The raw data required substantial cleanup. Of the 26,462 sets in the database, 6,983 (26.4%) '
        'had zero parts. These are non-buildable merchandise (keychains, bags, stationery, houseware) '
        'that must be filtered out. We also found 45,042 inventory records for 26,462 sets, meaning '
        'some sets have multiple inventory versions; we retained only version 1 (the primary inventory) '
        'to avoid duplicates. Within inventory_parts, we excluded spare parts (is_spare=True) since '
        'they are not part of the intended build. Missing values were minimal: only 7,170 missing image '
        'URLs (irrelevant for our features) and 2 colors without year data.'
    )
    pdf.p(
        'For theme mapping, we resolved 492 sub-themes to root themes using the hierarchical '
        'parent_id structure. We merged "Town" (1978-2004) into "City" (2005-present) as they '
        'represent the same product line. We selected 9 themes with >150 samples each, excluding '
        'non-buildable categories (Gear, Books, Collectible Minifigures) and themes too small for '
        'reliable cross-validation. Final dataset: 6,871 sets across 9 classes.'
    )
    pdf.s2('2.3 Feature Engineering')
    pdf.p(
        'For each set, we joined 7 tables and computed 256 features in 6 groups:'
    )
    pdf.tbl(
        ['Feature Group', 'Dims', 'Description'],
        [
            ['Color distribution', '166', 'Proportion of each color (quantity-weighted)'],
            ['Part category dist.', '70', 'Proportion of each part type (bricks, plates, etc.)'],
            ['Material dist.', '6', 'Proportion by material (Plastic, Rubber, etc.)'],
            ['Color statistics', '7', 'Weighted mean/std RGB, transparency ratio'],
            ['Scalar features', '5', 'num_parts, year, unique parts/colors/categories'],
            ['Minifig features', '2', 'Minifigure count and unique minifigure count'],
        ],
        cw=[38, 14, 118], fs=9
    )
    pdf.caption('Table 1: Feature groups (256 total dimensions).')

    pdf.p(
        'Color and part category distributions are quantity-weighted proportions summing to 1, '
        'capturing the compositional profile rather than absolute size. Only non-spare parts from '
        'the primary inventory (version 1) are included. Color statistics (avg/std RGB) provide a '
        'compact summary of the color palette; for instance, Star Wars sets have high grey values '
        'while Friends sets have high pink/purple values.'
    )

    pdf.s2('2.4 Dataset Examples')
    pdf.p(
        'Table 2 shows feature highlights for representative sets from different themes, '
        'illustrating how part composition reflects theme identity.'
    )
    pdf.tbl(
        ['Set', 'Theme', 'Parts', 'Minifigs', 'Top Color', 'Top Category'],
        [
            ['75192-1', 'Star Wars', '7541', '8', 'L.Bl.Gray (38%)', 'Plates (31%)'],
            ['42100-1', 'Technic', '2573', '0', 'D.Bl.Gray (26%)', 'Technic Pins (22%)'],
            ['10698-1', 'Creator', '790', '0', 'Yellow (12%)', 'Bricks (35%)'],
            ['10914-1', 'Duplo', '85', '3', 'Red (18%)', 'Duplo (89%)'],
            ['41395-1', 'Friends', '778', '4', 'White (16%)', 'Plates (26%)'],
            ['71741-1', 'Ninjago', '5685', '13', 'D.Bl.Gray (21%)', 'Plates (27%)'],
        ],
        cw=[22, 22, 16, 18, 42, 42], fs=9
    )
    pdf.caption('Table 2: Example sets showing distinctive composition per theme.')

    pdf.s2('2.5 Class Distribution')
    pdf.p(
        'The final dataset contains 6,871 LEGO sets across 9 themes. The distribution is imbalanced: '
        'City has 1,517 samples while Harry Potter has only 190 (8:1 ratio), providing a natural '
        'setting for our class balance experiments.'
    )
    pdf.fig('class_distribution.png', 'Figure 1: Samples per theme in the dataset.', w=125)

    # --- 3. Methods ---
    pdf.s1('3. Methods')
    pdf.s2('3.1 Models')
    pdf.p(
        'We evaluate three supervised learning models, of which at most one is deep learning-based:'
    )
    pdf.p(
        'Random Forest (RF): An ensemble of 300 decision trees with bootstrap aggregation and '
        'balanced class weights. RF handles high-dimensional sparse features well and provides '
        'interpretable feature importance scores. (scikit-learn RandomForestClassifier [1])'
    )
    pdf.p(
        'Histogram Gradient Boosting (HGB): A gradient boosting method using histogram-based '
        'binning for efficient training. Configured with 300 iterations and max_depth=8. HGB '
        'typically achieves state-of-the-art results on tabular data. (scikit-learn '
        'HistGradientBoostingClassifier [1])'
    )
    pdf.p(
        'Multi-Layer Perceptron (MLP): A neural network with three hidden layers (256, 128, 64 '
        'neurons), ReLU activation, and early stopping. This is our only deep learning model. '
        'Features are standardized (zero mean, unit variance) before training. (scikit-learn '
        'MLPClassifier [1])'
    )
    pdf.s2('3.2 Evaluation and Additional Methods')
    pdf.p(
        'All models are evaluated with 5-fold Stratified Cross-Validation, preserving class '
        'proportions in each fold. We report Accuracy and Macro-averaged Precision, Recall, and F1. '
        'Features are standardized within each fold to prevent data leakage. '
        'For experiments, we also use: PCA for dimensionality reduction [1], SMOTE for minority '
        'class oversampling [2,5], and t-SNE for 2D visualization [1].'
    )

    # --- 4. Results ---
    pdf.s1('4. Experiments and Results')
    pdf.s2('4.1 Baseline Performance')
    rows = []
    for name, res in baseline.items():
        m, s = res['mean'], res['std']
        rows.append([name, f"{m['accuracy']:.4f} +/- {s['accuracy']:.4f}",
                     f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                     f"{m['f1']:.4f} +/- {s['f1']:.4f}"])
    pdf.tbl(['Model', 'Accuracy', 'Prec', 'Rec', 'F1 (Macro)'], rows,
            cw=[42, 40, 25, 25, 38], fs=9)
    pdf.caption('Table 2: Baseline performance (5-fold CV).')

    pdf.p(
        'HGB achieves the best performance (95.5% accuracy, 0.939 F1), followed by RF (93.8%) and '
        'MLP (92.4%). Tree-based models outperforming MLP is consistent with literature on tabular data [3].'
    )

    # Per-class for best model
    pdf.s2('4.2 Per-Class Analysis')
    cm = np.array(baseline['Hist Gradient Boosting']['confusion_matrix'])
    pc_rows = []
    for i, cls in enumerate(class_names):
        tp, fp, fn = cm[i,i], cm[:,i].sum()-cm[i,i], cm[i,:].sum()-cm[i,i]
        pr = tp/(tp+fp) if tp+fp>0 else 0
        rc = tp/(tp+fn) if tp+fn>0 else 0
        f1 = 2*pr*rc/(pr+rc) if pr+rc>0 else 0
        pc_rows.append([cls, f'{pr:.3f}', f'{rc:.3f}', f'{f1:.3f}'])
    pdf.tbl(['Theme', 'Precision', 'Recall', 'F1'], pc_rows,
            cw=[42, 42, 42, 42], fs=9)
    pdf.caption('Table 3: Per-class metrics (HGB, best model).')

    pdf.p(
        'Duplo (F1=0.999) and Technic (0.988) are near-perfectly classified due to unique part systems. '
        'Harry Potter has lowest F1 (0.811) due to small sample size and overlap with other themes. '
        'Creator (0.893) is most confused with City and Star Wars, as it uses generic bricks.'
    )

    # Confusion matrix + t-SNE side concept — just show CM smaller
    pdf.fig('confusion_matrix_hist_gradient_boosting.png',
            'Figure 2: Normalized confusion matrix (HGB).', w=110)

    pdf.fig('tsne_visualization.png',
            'Figure 3: t-SNE visualization. Duplo (red), Technic (yellow-green), and Bionicle (blue) '
            'form distinct clusters; City/Creator/Star Wars overlap in the center.', w=115)

    # --- Experiment 1 ---
    pdf.s2('4.3 Exp 1: Training Data Amount')
    pdf.p(
        'We subsample training data from 20% to 100% within each CV fold.'
    )
    pdf.fig('learning_curves.png', 'Figure 4: Accuracy vs. training data fraction.', w=115)
    pdf.p(
        'Both models exceed 90% accuracy even at 20% data (~1,100 samples), demonstrating highly '
        'informative features. HGB consistently outperforms RF. Diminishing returns suggest the '
        'dataset size is sufficient.'
    )

    # --- Experiment 2 ---
    pdf.s2('4.4 Exp 2: Class Balance')
    pdf.p(
        'We compare: (a) no balancing, (b) class_weight="balanced", (c) SMOTE oversampling.'
    )
    exp2_rows = [[r['model'], r['strategy'], f"{r['accuracy']:.4f}", f"{r['f1']:.4f}"]
                 for r in experiments['exp2']]
    pdf.tbl(['Model', 'Strategy', 'Accuracy', 'F1'], exp2_rows,
            cw=[42, 45, 38, 38], fs=9)
    pdf.caption('Table 4: Class balance strategies.')
    pdf.p(
        'SMOTE improves RF F1 from 0.911 to 0.921 by boosting Harry Potter recall, but slightly hurts '
        'HGB (0.939 to 0.933). This shows that the best balancing strategy depends on the model.'
    )

    # --- Experiment 3 ---
    pdf.s2('4.5 Exp 3: PCA Dimensionality Reduction')
    pca_rows = [[r['label'], f"{r['accuracy']:.4f}"] for r in experiments['exp3_results']]
    pdf.tbl(['Components', 'Accuracy'], pca_rows, cw=[50, 50], fs=9)
    pdf.caption('Table 5: RF accuracy with PCA.')
    pdf.fig('pca_explained_variance.png',
            'Figure 5: Cumulative explained variance. 90% at 183 components, 95% at 205.', w=115)
    pdf.p(
        'PCA consistently hurts performance (best: 89.5% at 50 components vs 93.8% full). '
        'RF natively handles sparse high-dimensional features via feature subsampling at each split, '
        'making PCA redundant. The lost 10% variance contains discriminative signals.'
    )

    # --- Experiment 4 ---
    pdf.s2('4.6 Exp 4: Feature Ablation')
    exp4_rows = [[r['feature_group'], str(r['n_features']),
                  f"{r['accuracy']:.4f}", f"{r['f1']:.4f}"] for r in experiments['exp4']]
    pdf.tbl(['Feature Group', '#', 'Accuracy', 'F1'], exp4_rows,
            cw=[50, 15, 45, 45], fs=9)
    pdf.caption('Table 6: Feature ablation (RF).')
    pdf.fig('feature_ablation.png',
            'Figure 6: Accuracy and Macro F1 for each feature subset.', w=120)
    pdf.p(
        'Part categories are the single most informative group (88.0%), which makes sense: themes '
        'like Duplo, Technic, and Bionicle use fundamentally different part types. Color features '
        'achieve 82.1%, confirming that color palettes carry theme-specific signals (Friends uses '
        'pastels, Star Wars uses greys). Metadata alone reaches 80.8% with only 13 features, '
        'which is higher than expected. This is largely driven by the year feature. Combining '
        'all groups yields 93.8%, demonstrating strong complementarity.'
    )
    pdf.fig('feature_importance.png', 'Figure 7: Top 30 features by RF importance.', w=135)
    pdf.p(
        'The most important feature is the "Duplo, Quatro and Primo" part category, which perfectly '
        'identifies Duplo sets. Other top features include "Large Buildable Figures" (Bionicle), '
        'minifigure counts, Pearl Gold color, and the number of unique part categories.'
    )

    # --- Experiment 5 ---
    pdf.s2('4.7 Exp 5: Hyperparameter Sensitivity')
    pdf.fig('hyperparameter_sensitivity.png',
            'Figure 8: RF accuracy vs n_estimators (left) and max_depth (right).', w=140)
    pdf.p(
        'Accuracy saturates at ~200 trees (93.8%) and max_depth=20. Shallow trees (depth=5) '
        'significantly underperform (85.1%), indicating moderate decision boundary complexity.'
    )

    # --- Experiment 6 ---
    pdf.s2('4.8 Exp 6: Data Augmentation')
    pdf.p(
        'We apply data augmentation by duplicating training samples with added Gaussian noise '
        '(std = 0.01, 0.05, 0.1 on standardized features), effectively doubling the training set.'
    )
    exp6_rows = [[r['model'], r['label'], f"{r['accuracy']:.4f}", f"{r['f1']:.4f}"]
                 for r in experiments['exp6']]
    pdf.tbl(['Model', 'Augmentation', 'Accuracy', 'F1'], exp6_rows,
            cw=[42, 40, 40, 40], fs=9)
    pdf.caption('Table 7: Effect of data augmentation.')
    pdf.p(
        'Augmentation provides almost no benefit for RF and slightly hurts HGB. We believe this is '
        'because our features are compositional proportions that sum to 1, and adding Gaussian noise '
        'breaks this structure. Unlike image data where augmentation simulates realistic variations, '
        'adding noise to part distributions creates unrealistic compositions. For tabular data, '
        'augmentation strategies need to respect the feature semantics to be useful.'
    )

    # --- Experiment 7 ---
    pdf.s2('4.9 Exp 7: Year Feature Importance')
    pdf.p(
        'The year feature ranked highly in feature importance. To test whether models learn actual '
        'compositional patterns or simply memorize temporal associations, we remove year and re-evaluate.'
    )
    exp7_rows = [[r['model'], r['condition'], f"{r['accuracy']:.4f}", f"{r['f1']:.4f}"]
                 for r in experiments['exp7']]
    pdf.tbl(['Model', 'Condition', 'Accuracy', 'F1'], exp7_rows,
            cw=[42, 40, 40, 40], fs=9)
    pdf.caption('Table 8: Effect of removing the year feature.')
    pdf.fig('year_ablation.png',
            'Figure 9: Accuracy with and without the year feature.', w=120)
    pdf.p(
        'Removing year reduces RF accuracy by 0.7% and HGB by 0.7%, which confirms that year '
        'provides useful but not essential information. MLP is barely affected (0.08% drop), '
        'possibly because it relies less on any single feature. Since all models still achieve >93% '
        'accuracy without year, we can conclude that they learn genuine compositional patterns '
        'rather than just memorizing which themes existed in which era.'
    )

    # --- 5. Discussion ---
    pdf.s1('5. Discussion')
    pdf.s2('5.1 Expected vs. Observed Results')
    pdf.p(
        'Our results largely confirm the initial hypotheses. Themes with unique part systems, '
        'such as Duplo (large-format parts), Technic (beams, pins, gears), and Bionicle (constraction '
        'figures), are easily classified with F1 scores above 0.98. This validates our approach of using '
        'part category distributions as features, since these themes occupy entirely different regions '
        'of the part category space.'
    )
    pdf.p(
        'Among standard-brick themes, Friends (F1=0.968) is easier to classify than expected, '
        'likely because its distinctive pastel color palette (lavender, medium azure, bright pink) '
        'provides a strong signal. Creator (F1=0.893) is hardest, as expected, since it uses generic '
        'bricks that overlap with City and Star Wars. Tree-based models outperform MLP, consistent '
        'with recent benchmarks showing gradient boosting dominates neural networks on structured '
        'tabular data [3].'
    )
    pdf.p(
        'We did not expect PCA to hurt performance as much as it did. We initially thought some '
        'compression would help by removing noise, but our sparse feature vectors (most sets use '
        'fewer than 20 of 166 possible colors) seem to contain important discriminative signals '
        'in rare color dimensions that PCA discards.'
    )
    pdf.s2('5.2 Factors Affecting Results')
    pdf.p(
        'Several dataset characteristics significantly influence our results:'
    )
    pdf.p(
        '(1) Class imbalance: Harry Potter\'s small sample size (190 sets, 2.8% of data) results in '
        'the lowest recall across all models. SMOTE partially mitigates this for Random Forest '
        '(F1: 0.911 to 0.921) but not for HGB, suggesting that HGB\'s gradient-based learning '
        'already handles imbalance more gracefully.'
    )
    pdf.p(
        '(2) Feature sparsity: Most sets use fewer than 20 of the 166 possible colors, creating '
        'very sparse feature vectors. Tree-based models handle this naturally (each split examines '
        'one feature), while MLP may struggle with the curse of dimensionality.'
    )
    pdf.p(
        '(3) Temporal effects: The year feature captures theme-specific temporal patterns. '
        'Bionicle sets are concentrated in 2001-2016, while Ninjago started in 2011. '
        'The merged Town/City class spans 1978-2026, potentially introducing within-class variance '
        'due to evolving part design over decades.'
    )
    pdf.s2('5.3 Future Work')
    pdf.p(
        'Given more time, several extensions would be valuable: '
        '(1) Using set images from Rebrickable for CNN-based visual classification, comparing '
        'part-composition features vs. visual features. '
        '(2) Incorporating BrickLink secondary market price data for a regression task (predicting '
        'set value from composition). '
        '(3) Applying graph neural networks where parts are nodes and co-occurrence relationships '
        'are edges, potentially capturing spatial patterns. '
        '(4) Expanding to all 50+ themes with hierarchical classification. '
        '(5) Studying temporal drift: whether models trained on older sets generalize to newer ones.'
    )
    pdf.s2('5.4 Lessons Learned and Remaining Questions')
    pdf.p(
        'This project reinforced key insights about data-centric ML: '
        '(1) Feature engineering matters more than model selection. Our carefully designed features '
        'achieve 90%+ accuracy even with 20% of the data using a simple Random Forest. '
        '(2) Domain knowledge drives good features. Understanding that Duplo uses different parts, '
        'that Star Wars has a grey palette, and that Technic uses beams guided our design. '
        '(3) Standard techniques (PCA, SMOTE) can hurt when their assumptions don\'t match the data '
        'structure. Understanding when NOT to apply a technique is as important as knowing how.'
    )
    pdf.p(
        'We still have several open questions. First, why does PCA consistently degrade performance? '
        'Is this specifically due to the sparsity of our features, or would any linear dimensionality '
        'reduction fail here? It would be worth testing non-linear methods like autoencoders. '
        'Second, can our models generalize across time? If trained only on pre-2015 sets, would they '
        'accurately classify 2020+ sets where part designs have evolved? '
        'Third, Creator sets are hardest to classify because they use generic parts. Would adding '
        'part co-occurrence features (which specific parts tend to appear together) help distinguish '
        'Creator from other themes?'
    )

    # --- 6. References ---
    pdf.s1('6. References')
    pdf.set_font('NS', '', 10)
    refs = [
        '[1] Pedregosa et al. "Scikit-learn: ML in Python." JMLR 12 (2011): 2825-2830.',
        '[2] Lemaitre et al. "Imbalanced-learn: A Python Toolbox." JMLR 18(17) (2017): 1-5.',
        '[3] Grinsztajn et al. "Why do tree-based models still outperform deep learning on tabular data?" NeurIPS 2022.',
        '[4] Rebrickable.com. "LEGO Database." https://rebrickable.com/downloads/',
        '[5] Chawla et al. "SMOTE: Synthetic Minority Over-sampling." JAIR 16 (2002): 321-357.',
    ]
    for ref in refs:
        pdf.multi_cell(0, 5, ref)
        pdf.ln(1.5)

    body_pages = pdf.page_no()
    print(f"Main body: {body_pages} pages")

    # ============ APPENDIX ============
    pdf.add_page()
    pdf.is_appendix = True
    pdf.s1('Appendix: Program Code')
    pdf.p('All code written in Python 3 using scikit-learn, pandas, numpy, matplotlib.')

    code_files = [
        ('feature_engineering.py', 'Build dataset from raw CSV'),
        ('train_evaluate.py', 'Model training and CV evaluation'),
        ('experiments.py', 'All 5 experiments'),
        ('visualize.py', 'Generate report figures'),
    ]

    for filename, desc in code_files:
        pdf.s2(f'{filename}')
        pdf.set_font('NS', 'I', 9)
        pdf.cell(0, 5, desc, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(1)
        filepath = os.path.join(BASE_DIR, filename)
        with open(filepath, 'r') as f:
            code = f.read()
        pdf.set_font('NM', '', 6.5)
        for line in code.split('\n'):
            if len(line) > 120:
                line = line[:117] + '...'
            pdf.cell(0, 2.8, line, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(3)

    output_path = os.path.join(BASE_DIR, '312554027_report.pdf')
    pdf.output(output_path)
    print(f'Report saved to {output_path}')
    print(f'Total pages: {pdf.page_no()} (body: {body_pages}, appendix: {pdf.page_no()-body_pages})')


if __name__ == '__main__':
    build_report()
