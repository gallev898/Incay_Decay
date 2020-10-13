import torch

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import get_args

if __name__ == '__main__':
    args = get_args()
    per_sample_collector = torch.load('/Users/gallevshalev/Desktop/incay_decay_trained_models/{}/per_sample_collector'.format(args.runname))


    class dataset_map:
        confidence = [np.average(x[1]['conf']) for i, x in enumerate(per_sample_collector.items())]
        variability = [np.std(x[1]['pred']) for i, x in enumerate(per_sample_collector.items())]
        correctness = [sum(x[1]['pred']) / 60 for i, x in enumerate(per_sample_collector.items())]

    dataset_map = dataset_map()

    # Plot
    _, ax = plt.subplots(figsize=(9, 7))

    sns.scatterplot(x=dataset_map.variability, y=dataset_map.confidence, hue=dataset_map.correctness, ax=ax)
    sns.kdeplot(data=dataset_map.variability, data2=dataset_map.confidence,
                levels=8, color=sns.color_palette("Paired")[7], linewidths=1, ax=ax)

    ax.set(title='Data map for {}'.format(args.runname),
           xlabel='Variability', ylabel='Confidence')

    # Annotations
    box_style = {'boxstyle': 'round', 'facecolor': 'white', 'ec': 'black'}
    ax.text(0.14, 0.84,
            'easy-to-learn',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=box_style)
    ax.text(0.75, 0.5,
            'ambiguous',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=box_style)
    ax.text(0.14, 0.14,
            'hard-to-learn',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=box_style)

    ax.legend(title='Correctness')

    plt.show()
    d = 0
