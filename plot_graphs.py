import torch

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import *


def plot_data_map(examples, plot_name):
    class dataset_map():
        def __init__(self, examples):
            self.confidence = [max(x[1]['cosine'][30:]) - min(x[1]['cosine'][30:]) for i, x in
                               enumerate(examples.items())]
            self.variability = [max(x[1]['norm'][30:]) - min(x[1]['norm'][30:]) for i, x in enumerate(examples.items())]
            self.correctness = [sum(x[1]['pred'][30:]) / 30 for i, x in enumerate(examples.items())]

    dataset_map = dataset_map(examples)

    # Plot
    _, ax = plt.subplots(figsize=(9, 7))

    sns.scatterplot(x=dataset_map.variability, y=dataset_map.confidence, hue=dataset_map.correctness, ax=ax)
    sns.kdeplot(data=dataset_map.variability, data2=dataset_map.confidence,
                levels=8, color=sns.color_palette("Paired")[7], linewidths=1, ax=ax)

    ax.set(title='Plot_Name: {} Model {}'.format(plot_name, args.runname),
           xlabel='Variability', ylabel='Confidence')

    ax.legend(title='Correctness')

    plt.show()


def select_examples_by_condition(per_sample_collector, condition=None):
    if condition is None:
        return per_sample_collector

    return {k: v for k, v in per_sample_collector.items() if condition(v)}


def plot_points_by_keys(keys, wrong_examples, correct_examples, num_of_exmaples):
    correct_examples = list(correct_examples.items())[:num_of_exmaples]
    wrong_examples = list(wrong_examples.items())[:num_of_exmaples]

    for key in keys:

        correct_examples_key = [x[1][key] for x in correct_examples]
        wrong_examples_key = [x[1][key] for x in wrong_examples]

        for i in range(num_of_exmaples):
            plt.plot(np.arange(60), correct_examples_key[i], color='green')
            plt.plot(np.arange(60), wrong_examples_key[i], color='red')

        plt.title(key)
        plt.show()
        plt.clf()


def consistently_classified_through_epochs(per_sample_collector):
    start_idx = 0
    correct_examples_lens, wrong_examples_lens = [], []
    for i in range(1, 60):
        end_idx = i
        correct_examples_lens.append(
            len(select_examples_by_condition(per_sample_collector, lambda x: sum(x['pred'][start_idx:end_idx]) == end_idx)))
        wrong_examples_lens.append(
            len(select_examples_by_condition(per_sample_collector, lambda x: sum(x['pred'][start_idx:end_idx]) == 0)))

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(correct_examples_lens)) - 0.2, correct_examples_lens,
           width=0.2, color='g', align='center')
    ax.bar(np.arange(len(wrong_examples_lens)) - 0.2, wrong_examples_lens,
           width=0.2, color='r', align='center')
    ax.set_title('Correct vs Wrong consistently classified until epoch i')
    plt.show()


if __name__ == '__main__':
    args = get_args()

    print('Loading PATHS.COLLECTOR_PATH from :{}'.format(args.runname))
    per_sample_collector = torch.load( PATHS.COLLECTOR_PATH.value.format(args.runname))

    # consistently_classified_through_epochs(per_sample_collector)
