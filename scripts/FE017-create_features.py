import pandas as pd
import numpy as np
import os
from tqdm import tqdm

feature_files = [f for f in os.listdir('../data/FE016/')]
for f in tqdm(feature_files):
    results_files = [f for f in os.listdir('../data/FE017')]
    if f.replace('FE016','FE017') in results_files:
        print('File {} already exists, skipping'.format(f.replace('FE016','FE017')))
        continue

    df = pd.read_parquet(f'../data/FE016/{f}')
    dist_cols = [x for x in df.columns if 'distance_' in x]
    dist_to_0_cols = [x for x in dist_cols if 'to_0' in x]
    dist_to_1_cols = [x for x in dist_cols if 'to_1' in x]
    df['dist_to_0_mean'] = df[dist_to_0_cols].mean(axis=1)
    df['dist_to_1_mean'] = df[dist_to_1_cols].mean(axis=1)
    df['dist_to_0_min'] = df[dist_to_0_cols].min(axis=1)
    df['dist_to_1_min'] = df[dist_to_1_cols].min(axis=1)
    df['dist_to_0_max'] = df[dist_to_0_cols].max(axis=1)
    df['dist_to_1_max'] = df[dist_to_1_cols].max(axis=1)
    df['dist_to_0_std'] = df[dist_to_0_cols].std(axis=1)
    df['dist_to_1_std'] = df[dist_to_1_cols].std(axis=1)

    df['dist_to_0_mean'] = df[dist_to_0_cols].mean(axis=1)
    df['dist_to_1_mean'] = df[dist_to_1_cols].mean(axis=1)
    df['dist_to_0_min'] = df[dist_to_0_cols].min(axis=1)
    df['dist_to_1_min'] = df[dist_to_1_cols].min(axis=1)
    df['dist_to_0_max'] = df[dist_to_0_cols].max(axis=1)
    df['dist_to_1_max'] = df[dist_to_1_cols].max(axis=1)
    df['dist_to_0_std'] = df[dist_to_0_cols].std(axis=1)
    df['dist_to_1_std'] = df[dist_to_1_cols].std(axis=1)

    df['dist_to_0_mean'] = df[dist_to_0_cols].mean(axis=1)
    df['dist_to_1_mean'] = df[dist_to_1_cols].mean(axis=1)
    df['dist_to_0_min'] = df[dist_to_0_cols].min(axis=1)
    df['dist_to_1_min'] = df[dist_to_1_cols].min(axis=1)
    df['dist_to_0_max'] = df[dist_to_0_cols].max(axis=1)
    df['dist_to_1_max'] = df[dist_to_1_cols].max(axis=1)
    df['dist_to_0_std'] = df[dist_to_0_cols].std(axis=1)
    df['dist_to_1_std'] = df[dist_to_1_cols].std(axis=1)

    valence_cols = [x for x in df.columns if '_valence' in x]
    val_to_0_cols = [x for x in valence_cols if 'to_0' in x]
    val_to_1_cols = [x for x in valence_cols if 'to_1' in x]
    df['val_not_0_mean'] = df[val_to_0_cols].mean(axis=1)
    df['val_not_1_mean'] = df[val_to_1_cols].mean(axis=1)
    df['val_not_0_max'] = df[val_to_0_cols].max(axis=1)
    df['val_not_1_max'] = df[val_to_1_cols].max(axis=1)
    df['val_not_0_min'] = df[val_to_0_cols].min(axis=1)
    df['val_not_1_min'] = df[val_to_1_cols].min(axis=1)
    df['val_not_0_std'] = df[val_to_0_cols].std(axis=1)
    df['val_not_1_std'] = df[val_to_1_cols].std(axis=1)

    atomic_mass_cols = [x for x in df.columns if '_atomic_mass' in x]
    atomic_mass_to_0_cols = [x for x in atomic_mass_cols if 'to_0' in x]
    atomic_mass_to_1_cols = [x for x in atomic_mass_cols if 'to_1' in x]
    df['atomic_mass_not_0_mean'] = df[atomic_mass_to_0_cols].mean(axis=1)
    df['atomic_mass_not_1_mean'] = df[atomic_mass_to_1_cols].mean(axis=1)
    df['atomic_mass_not_0_max'] = df[atomic_mass_to_0_cols].max(axis=1)
    df['atomic_mass_not_1_max'] = df[atomic_mass_to_1_cols].max(axis=1)
    df['atomic_mass_not_0_min'] = df[atomic_mass_to_0_cols].min(axis=1)
    df['atomic_mass_not_1_min'] = df[atomic_mass_to_1_cols].min(axis=1)
    df['atomic_mass_not_0_std'] = df[atomic_mass_to_0_cols].std(axis=1)
    df['atomic_mass_not_1_std'] = df[atomic_mass_to_1_cols].std(axis=1)

    for col in dist_cols:
        df[col]
        temp = df[col]**3
        df[col+'_cube_inverse'] = temp**-1
    relative_col_names = [x.strip('distance')[1:] for x in dist_cols]
    for r in relative_col_names:
        df[f'{r}_atomic_mass_x_cube_inv_dist'] = df[f'distance_{r}_cube_inverse'] * df[f'{r}_atomic_mass']
        df[f'{r}_valence_x_cube_inv_dist'] = df[f'distance_{r}_cube_inverse'] * df[f'{r}_valence']
        df[f'{r}_spin_multiplicity_x_cube_inv_dist'] = df[f'distance_{r}_cube_inverse'] * df[f'{r}_spin_multiplicity']
    # Save results
    df.to_parquet('../data/FE017/{}'.format(f.replace('FE016','FE017')))

