# LEGO Set Theme Classification Dataset

## Research Question

Can we predict which LEGO product theme (e.g., Star Wars, Technic, Duplo) a set belongs to, based solely on its part composition — the distribution of colors, part categories, materials, and structural features?

## Data Source

- **Source**: Rebrickable.com (https://rebrickable.com/downloads/)
- **License**: Rebrickable data is provided under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
- **Raw data tables used**: sets, themes, inventories, inventory_parts, inventory_minifigs, colors, parts, part_categories

## Dataset Construction

### Process

1. **Downloaded** 9 raw CSV tables from Rebrickable's public database dumps (1.49M+ inventory part records, 26,462 sets, 492 themes, etc.)
2. **Mapped** all sub-themes to their root theme using the hierarchical theme structure (e.g., "Star Wars Episode IV" → "Star Wars")
3. **Merged** the "Town" theme (1978-2004) into "City" (2005-present), as they are the same product line
4. **Filtered** to 9 target themes and removed non-buildable merchandise (sets with 0 parts)
5. **Joined** 7 tables (sets → inventories → inventory_parts → parts/colors/part_categories + inventory_minifigs) to compute per-set feature vectors
6. **Engineered** 256 features across 6 feature groups

### Feature Groups

| Group | Dimensions | Description |
|-------|-----------|-------------|
| Color distribution | 166 | Proportion of each color (quantity-weighted) |
| Part category distribution | 70 | Proportion of each part category (bricks, plates, tiles, etc.) |
| Material distribution | 6 | Proportion of each material type (Plastic, Rubber, etc.) |
| Color statistics | 7 | Weighted mean/std of RGB values, proportion of transparent parts |
| Scalar features | 5 | num_parts, year, num_unique_parts/colors/categories |
| Minifig features | 2 | num_minifigs, num_unique_minifigs |

### Class Distribution

| Theme | Samples |
|-------|---------|
| City (incl. Town) | 1,517 |
| Duplo | 1,377 |
| Star Wars | 977 |
| Ninjago | 646 |
| Friends | 639 |
| Creator | 592 |
| Technic | 564 |
| Bionicle | 369 |
| Harry Potter | 190 |
| **Total** | **6,871** |

### Data Quality

- No missing values in the final dataset
- All features are numeric (continuous proportions or counts)
- Class imbalance ratio (largest/smallest): ~8:1

## Files

- `output/dataset.csv` — Final feature matrix (6,871 rows × 258 columns, including set_num and theme label)
- `output/dataset_info.txt` — Summary statistics
- `data/` — Raw CSV files from Rebrickable

## Software Used

- Python 3.x with pandas, numpy, scikit-learn
- Data downloaded via Rebrickable public CSV dumps (https://cdn.rebrickable.com/media/downloads/)
