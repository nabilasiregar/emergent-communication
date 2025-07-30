import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('analysis/human_compositionality.csv')

if 'max_len' not in df.columns:
    df['max_len'] = df['seed_folder'].str.extract(r'maxlen(\d+)').astype(int)

metrics = ['TopSim', 'PosDis', 'TRE']
agg = (
    df
    .groupby(['max_len', 'type'])[metrics]
    .agg(['mean', 'std'])
    .sort_index()
)

max_lens = sorted(df['max_len'].unique())
types    = ['actual', 'random']
colors   = {'actual': '#66b2ff', 'random': '#888888'}
width    = 0.35

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
fig.subplots_adjust(left=0.10, right=0.95, bottom=0.25, wspace=0.01)

for idx, (ax, metric) in enumerate(zip(axes, metrics)):
    for i, t in enumerate(types):
        means = [agg.loc[(ml, t), (metric, 'mean')] for ml in max_lens]
        stds  = [agg.loc[(ml, t), (metric, 'std')]  for ml in max_lens]
        y = np.arange(len(max_lens)) + (i - 0.5) * width

        ax.barh(
            y, means, xerr=stds,
            height=width, align='center',
            color=colors[t], ecolor='black', capsize=3,
            label='Emergent' if t=='actual' else 'Random'

        )

    ax.set_title(metric)
    if idx == 0:
        ax.set_yticks(np.arange(len(max_lens)))
        ax.set_yticklabels([f'maxlen={ml}' for ml in max_lens])
        # ax.set_ylabel('Score')
    else:
        ax.set_yticks([])
        ax.tick_params(axis='y', left=False)

    ax.invert_yaxis()
    ax.set_box_aspect(1)

    if metric == 'TRE':
        ax.invert_xaxis()

axes[0].set_xlabel('Score')
axes[1].set_xlabel('Score')
axes[2].set_xlabel('TRE')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    title='Type',
    ncol=2,
    frameon=False,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.01)
)

plt.show()
