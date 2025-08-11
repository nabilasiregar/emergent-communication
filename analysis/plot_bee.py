import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

df = pd.read_csv('results/bee_compositionality.csv')
seeds = {27, 31, 123, 2025, 42}
df['seed'] = df['seed'].str.extract(r'seed(\d+)$').astype(int)
df = df[df['seed'].isin(seeds)]

protocols = [
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
    'hop_count_distance_vector_sum_direction': 'step count\n(direction via vector sum)',
    'sum_distances_vector_sum_direction': 'total distance\n(direction via vector sum)',
    'hop_count_distance_angle_direction': 'step count\n(direction via angle)',
    'sum_distances_angle_direction': 'total distance\n(direction via angle)',
    'random': 'random'
}

main_plots = [
    ('TopSim', 'topsim_rank'),
    ('PosDis (total)', 'posdis_total'),
    ('PosDis (distance)', 'posdis_distance'),
    ('PosDis (direction)', 'posdis_direction'),
    ('TRE (both weights)', 'tre')
]

variant_plots = [
    ('TRE (distance only)', 'tre_dist_only'),
    ('TRE (direction only)', 'tre_dir_only'),
    ('TRE (distance heavy)', 'tre_dist_heavy'),
    ('TRE (direction heavy)', 'tre_dir_heavy'),
]

grouped = df.groupby(['seed', 'hypothesis']).mean().reset_index()
stats = grouped.groupby('hypothesis').agg(['mean', 'std'])

fig = plt.figure(figsize=(24, 10))
gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1], width_ratios=[1,1,1,1,1], hspace=0.4, wspace=0.35)

for i, (title, col) in enumerate(main_plots):
    ax = fig.add_subplot(gs[0, i])
    prots = protocols
    means = [stats.loc[p, (col, 'mean')] for p in prots]
    stds = [stats.loc[p, (col, 'std')] for p in prots]
    y_pos = np.arange(len(prots))
    ax.barh(y_pos, means, xerr=stds, align='center', color='#FF9933', ecolor='black', capsize=3)
    ax.invert_yaxis()
    if i == 0:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([readable[p] for p in prots], fontsize=16)
        ax.set_ylabel('protocol', fontsize=16)
    else:
        ax.set_yticks([])
        ax.tick_params(axis='y', left=False)
    ax.set_title(title, pad=8, fontsize=16)
    ax.tick_params(axis='x', labelsize=16)

for k, (title, col) in enumerate(variant_plots):
    ax = fig.add_subplot(gs[1, k])
    prots = proto_no_rand
    means = [stats.loc[p, (col, 'mean')] for p in prots]
    stds = [stats.loc[p, (col, 'std')] for p in prots]
    y_pos = np.arange(len(prots))
    ax.barh(y_pos, means, xerr=stds, align='center', color='#FF9933', ecolor='black', capsize=3)
    ax.invert_yaxis()
    if k == 0:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([readable[p] for p in prots], fontsize=16)
        ax.set_ylabel('protocol', fontsize=16)
    else:
        ax.set_yticks([])
        ax.tick_params(axis='y', left=False)
    ax.set_title(title, pad=8, fontsize=16)
    ax.tick_params(axis='x', labelsize=16)

plt.subplots_adjust(top=0.95, bottom=0.07)
fig.savefig("bee_compositionality.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
