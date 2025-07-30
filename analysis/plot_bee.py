import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('analysis/bee_compositionality.csv')
seeds = {27, 31, 123, 2025, 42}
df['seed'] = df['seed'].str.extract(r'seed(\d+)$').astype(int)
df = df[df['seed'].isin(seeds)]

protocols     = [
    'coordinates',
    'hop_count_distance_vector_sum_direction',
    'sum_distances_vector_sum_direction',
    'hop_count_distance_angle_direction',
    'sum_distances_angle_direction',
    'random'
]
proto_no_rand = protocols[:-1]

readable = {
    'coordinates': 'coordinates',
    'hop_count_distance_vector_sum_direction': 'hop count\n(direction via vector sum)',
    'sum_distances_vector_sum_direction': 'total distance\n(direction via vector sum)',
    'hop_count_distance_angle_direction': 'hop count\n(direction via angle)',
    'sum_distances_angle_direction': 'total distance\n(direction via angle)',
    'random': 'random'
}

main_plots = [
    ('TopSim (lin)',       'topsim_lin'),
    ('TopSim (log)',       'topsim_log'),
    ('TopSim (rank)',      'topsim_rank'),
    ('PosDis (total)',     'posdis_total'),
    ('PosDis (distance)',  'posdis_distance'),
    ('PosDis (direction)', 'posdis_direction'),
    ('TRE (both weights)', 'tre')
]

variant_plots = [
    ('TRE (distance only)',   'tre_dist_only'),
    ('TRE (direction only)',  'tre_dir_only'),
    ('TRE (distance heavy)',  'tre_dist_heavy'),
    ('TRE (direction heavy)', 'tre_dir_heavy'),
]

grouped = df.groupby(['seed', 'hypothesis']).mean().reset_index()
stats   = grouped.groupby('hypothesis').agg(['mean', 'std'])

fig, axes = plt.subplots(3, 4,
                         figsize=(4*4 + 2, 3*3),
                         constrained_layout=True)
axes = axes.flatten()

for idx, (title, col) in enumerate(main_plots):
    ax = axes[idx]
    prots = protocols
    means = [stats.loc[p, (col, 'mean')] for p in prots]
    stds  = [stats.loc[p, (col, 'std')]  for p in prots]
    y_pos = np.arange(len(prots))

    ax.barh(y_pos, means, xerr=stds,
            align='center', color='#FF9933',
            ecolor='black', capsize=3)

    if idx % 4 == 0:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([readable[p] for p in prots])
        ax.set_ylabel('protocol')
    else:
        ax.set_yticks([])
        ax.tick_params(axis='y', left=False)

    ax.set_title(title, pad=8)
    ax.invert_yaxis()
    ax.set_box_aspect(1)

axes[7].axis('off')

for j, (title, col) in enumerate(variant_plots):
    ax = axes[8 + j]
    prots = proto_no_rand
    means = [stats.loc[p, (col, 'mean')] for p in prots]
    stds  = [stats.loc[p, (col, 'std')]  for p in prots]
    y_pos = np.arange(len(prots))

    ax.barh(y_pos, means, xerr=stds,
            align='center', color='#FF9933',
            ecolor='black', capsize=3)

    if (8 + j) % 4 == 0:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([readable[p] for p in prots])
        ax.set_ylabel('protocol')
    else:
        ax.set_yticks([])
        ax.tick_params(axis='y', left=False)

    ax.set_title(title, pad=8)
    ax.invert_yaxis()
    ax.set_box_aspect(1)

plt.show()

