"""
Feature Engineering: Build LEGO theme classification dataset from Rebrickable raw data.

Reads raw CSV files from data/, joins tables, computes per-set features
(color distribution, part category distribution, material distribution,
color statistics, minifig counts), and outputs a single dataset.csv.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

# 9 target themes (Town merged into City)
TARGET_THEMES = [
    'Star Wars', 'Friends', 'Ninjago', 'Technic',
    'Bionicle', 'City', 'Creator', 'Duplo', 'Harry Potter'
]

# Root theme names that should be merged into City
CITY_MERGE = {'City', 'Town'}


def load_csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, f'{name}.csv'))


def build_theme_map(themes_df):
    """Map every theme_id to its root theme name."""
    theme_dict = themes_df.set_index('id')[['name', 'parent_id']].to_dict('index')
    cache = {}

    def get_root(tid):
        if tid in cache:
            return cache[tid]
        if tid not in theme_dict:
            return None
        info = theme_dict[tid]
        pid = info['parent_id']
        if pd.isna(pid):
            cache[tid] = info['name']
        else:
            cache[tid] = get_root(int(pid))
        return cache[tid]

    for tid in theme_dict:
        get_root(tid)
    return cache


def hex_to_rgb(hex_str):
    """Convert hex color string to (R, G, B) tuple."""
    try:
        h = hex_str.lstrip('#')
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except (ValueError, TypeError, AttributeError):
        return (128, 128, 128)


def run():
    """Main pipeline: load raw CSVs -> filter -> join -> compute features -> save dataset."""
    print("=== Feature Engineering ===")

    # --- Load all 9 raw CSV tables from Rebrickable ---
    themes = load_csv('themes')       # 492 themes with hierarchical parent_id
    sets = load_csv('sets')           # 26,462 sets with year, theme_id, num_parts
    inventories = load_csv('inventories')  # Maps sets to inventory IDs (may have multiple versions)
    inv_parts = load_csv('inventory_parts')    # 1.49M rows: detailed parts per inventory
    inv_minifigs = load_csv('inventory_minifigs')  # Minifigures per inventory
    colors = load_csv('colors')       # 275 colors with RGB hex values and transparency flag
    parts = load_csv('parts')         # 61K parts with category and material
    part_cats = load_csv('part_categories')  # 76 part categories (Bricks, Plates, Technic, etc.)

    # --- Step 1: Map themes and filter sets ---
    theme_map = build_theme_map(themes)
    sets['root_theme'] = sets['theme_id'].map(theme_map)

    # Merge Town into City
    sets['root_theme'] = sets['root_theme'].replace('Town', 'City')

    # Filter: num_parts > 0 and target themes only
    sets_filtered = sets[(sets['num_parts'] > 0) & (sets['root_theme'].isin(TARGET_THEMES))].copy()
    print(f"Sets after filtering: {len(sets_filtered)}")
    print(f"Class distribution:\n{sets_filtered['root_theme'].value_counts().to_string()}\n")

    # --- Step 2: Join to inventories (version 1 only) ---
    inv_v1 = inventories[inventories['version'] == 1][['id', 'set_num']]
    sets_inv = sets_filtered.merge(inv_v1, on='set_num', how='inner')
    sets_inv = sets_inv.rename(columns={'id': 'inventory_id'})
    print(f"Sets with inventory: {len(sets_inv)}")

    # --- Step 3: Compute part-based features ---
    # Filter spare parts
    inv_parts_used = inv_parts[inv_parts['is_spare'] == False][['inventory_id', 'part_num', 'color_id', 'quantity']]

    # Keep only inventories in our set
    valid_inv_ids = set(sets_inv['inventory_id'])
    inv_parts_used = inv_parts_used[inv_parts_used['inventory_id'].isin(valid_inv_ids)]
    print(f"Inventory part rows for our sets: {len(inv_parts_used)}")

    # Join part info (category, material)
    parts_info = parts[['part_num', 'part_cat_id', 'part_material']]
    inv_parts_enriched = inv_parts_used.merge(parts_info, on='part_num', how='left')

    # Join color info (rgb, is_trans)
    colors_info = colors[['id', 'rgb', 'is_trans']].rename(columns={'id': 'color_id'})
    inv_parts_enriched = inv_parts_enriched.merge(colors_info, on='color_id', how='left')

    # Parse RGB values
    rgb_values = inv_parts_enriched['rgb'].apply(hex_to_rgb)
    inv_parts_enriched['r'] = rgb_values.apply(lambda x: x[0])
    inv_parts_enriched['g'] = rgb_values.apply(lambda x: x[1])
    inv_parts_enriched['b'] = rgb_values.apply(lambda x: x[2])

    # --- 3a. Color distribution (275 dims) ---
    print("Computing color distribution...")
    color_pivot = inv_parts_enriched.groupby(['inventory_id', 'color_id'])['quantity'].sum().unstack(fill_value=0)
    # Normalize each row to proportions
    color_totals = color_pivot.sum(axis=1)
    color_dist = color_pivot.div(color_totals, axis=0)
    color_dist.columns = [f'color_{c}' for c in color_dist.columns]

    # --- 3b. Part category distribution (76 dims) ---
    print("Computing part category distribution...")
    cat_pivot = inv_parts_enriched.groupby(['inventory_id', 'part_cat_id'])['quantity'].sum().unstack(fill_value=0)
    cat_totals = cat_pivot.sum(axis=1)
    cat_dist = cat_pivot.div(cat_totals, axis=0)
    cat_dist.columns = [f'cat_{c}' for c in cat_dist.columns]

    # --- 3c. Material distribution ---
    print("Computing material distribution...")
    mat_pivot = inv_parts_enriched.groupby(['inventory_id', 'part_material'])['quantity'].sum().unstack(fill_value=0)
    mat_totals = mat_pivot.sum(axis=1)
    mat_dist = mat_pivot.div(mat_totals, axis=0)
    mat_dist.columns = [f'mat_{c}' for c in mat_dist.columns]

    # --- 3d. Color statistics ---
    print("Computing color statistics...")
    def weighted_color_stats(group):
        q = group['quantity'].values.astype(float)
        total_q = q.sum()
        if total_q == 0:
            return pd.Series({'avg_r': 128, 'avg_g': 128, 'avg_b': 128,
                              'std_r': 0, 'std_g': 0, 'std_b': 0, 'prop_transparent': 0})
        weights = q / total_q
        r, g, b = group['r'].values, group['g'].values, group['b'].values
        avg_r = np.average(r, weights=weights)
        avg_g = np.average(g, weights=weights)
        avg_b = np.average(b, weights=weights)
        std_r = np.sqrt(np.average((r - avg_r) ** 2, weights=weights))
        std_g = np.sqrt(np.average((g - avg_g) ** 2, weights=weights))
        std_b = np.sqrt(np.average((b - avg_b) ** 2, weights=weights))
        trans = group['is_trans'].values.astype(float)
        prop_trans = np.average(trans, weights=weights)
        return pd.Series({'avg_r': avg_r, 'avg_g': avg_g, 'avg_b': avg_b,
                          'std_r': std_r, 'std_g': std_g, 'std_b': std_b,
                          'prop_transparent': prop_trans})

    color_stats = inv_parts_enriched.groupby('inventory_id').apply(weighted_color_stats, include_groups=False)

    # --- 3e. Scalar features ---
    print("Computing scalar features...")
    scalar_feats = inv_parts_enriched.groupby('inventory_id').agg(
        num_unique_parts=('part_num', 'nunique'),
        num_unique_colors=('color_id', 'nunique'),
        num_unique_categories=('part_cat_id', 'nunique'),
    )

    # --- Step 4: Minifig features ---
    print("Computing minifig features...")
    inv_minifigs_filtered = inv_minifigs[inv_minifigs['inventory_id'].isin(valid_inv_ids)]
    minifig_feats = inv_minifigs_filtered.groupby('inventory_id').agg(
        num_minifigs=('quantity', 'sum'),
        num_unique_minifigs=('fig_num', 'nunique'),
    )

    # --- Step 5: Assemble final dataset ---
    print("Assembling final dataset...")
    # Start with set info
    dataset = sets_inv[['set_num', 'inventory_id', 'root_theme', 'num_parts', 'year']].copy()
    dataset = dataset.set_index('inventory_id')

    # Join all feature groups
    dataset = dataset.join(color_dist, how='left')
    dataset = dataset.join(cat_dist, how='left')
    dataset = dataset.join(mat_dist, how='left')
    dataset = dataset.join(color_stats, how='left')
    dataset = dataset.join(scalar_feats, how='left')
    dataset = dataset.join(minifig_feats, how='left')

    # Fill NaN (sets with no minifigs get 0)
    dataset['num_minifigs'] = dataset['num_minifigs'].fillna(0).astype(int)
    dataset['num_unique_minifigs'] = dataset['num_unique_minifigs'].fillna(0).astype(int)
    dataset = dataset.fillna(0)

    # Rename label column
    dataset = dataset.rename(columns={'root_theme': 'theme'})
    dataset = dataset.reset_index(drop=True)

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'dataset.csv')
    dataset.to_csv(output_path, index=False)

    # Summary
    feature_cols = [c for c in dataset.columns if c not in ['set_num', 'theme']]
    summary = (
        f"Dataset shape: {dataset.shape}\n"
        f"Feature count: {len(feature_cols)}\n"
        f"Samples per class:\n{dataset['theme'].value_counts().to_string()}\n\n"
        f"Feature groups:\n"
        f"  Color distribution: {sum(1 for c in feature_cols if c.startswith('color_'))}\n"
        f"  Part category distribution: {sum(1 for c in feature_cols if c.startswith('cat_'))}\n"
        f"  Material distribution: {sum(1 for c in feature_cols if c.startswith('mat_'))}\n"
        f"  Color statistics: 7 (avg/std RGB + prop_transparent)\n"
        f"  Scalar features: 5 (num_parts, year, num_unique_parts/colors/categories)\n"
        f"  Minifig features: 2 (num_minifigs, num_unique_minifigs)\n"
    )
    print(summary)

    info_path = os.path.join(OUTPUT_DIR, 'dataset_info.txt')
    with open(info_path, 'w') as f:
        f.write(summary)

    print(f"Dataset saved to {output_path}")
    print(f"Summary saved to {info_path}")
    return dataset


if __name__ == '__main__':
    run()
