from utils import dataset_helpers, input_helpers, label_helpers, mask_helpers, geofiles, metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def timeseries_length_comparison(dataset: str, numeric_names: bool = False):
    labels, n_clear_sar, n_clear_fusion = [], [], []
    aoi_ids = dataset_helpers.get_aoi_ids(dataset)
    for aoi_id in tqdm(aoi_ids):
        metadata = dataset_helpers.metadata(dataset)['aois'][aoi_id]
        n_clear_sar.append(len([_ for _, _, _, s1, s2 in metadata if s1]))
        n_clear_fusion.append(len([_ for _, _, _, s1, s2 in metadata if s1 and s2]))
        labels.append(aoi_id)

    if numeric_names:
        labels = [f'AOI {i + 1}' for i in range(len(labels))]
    else:
        labels = [aoi_id[4:15] for aoi_id in labels]
    width = 0.2
    inbetween_space = 0.1

    fontsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(len(aoi_ids), 6))

    center_pos = np.arange(len(aoi_ids))
    offset = (width + inbetween_space) / 2
    ax.bar(center_pos - offset, n_clear_sar, width, label='S1', color='#1f77b4')
    ax.bar(center_pos + offset, n_clear_fusion, width, label='S1 + S2', color='#d62728')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_xlim((-0.5, len(labels) - 0.5))

    max_value = (max(n_clear_sar) // 5 + 1) * 5
    y_ticks = np.arange(0, max_value + 1, 5)
    ax.set_ylim((0, max_value))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.0f}' for y_tick in y_ticks], fontsize=fontsize)
    ax.set_ylabel('Timeseries length', fontsize=fontsize)

    ax.legend(ncol=2, handletextpad=0.4, columnspacing=1.2, frameon=False, loc='upper center', fontsize=fontsize)

    plt.show()


def urban_extraction_comparison(config_names: list, dataset: str, numeric_names: bool = False):

    width = 0.2
    inbetween_space = 0.1
    fontsize = 20
    colors = ['#1f77b4', '#d62728']
    labels = ['S1', 'S1 + S2']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    aoi_ids = dataset_helpers.get_aoi_ids(dataset)
    fig, ax = plt.subplots(1, 1, figsize=(len(aoi_ids), 6))

    for i, config_name in enumerate(config_names):
        data = []
        for aoi_id in tqdm(aoi_ids):
            metadata = dataset_helpers.metadata(dataset)['aois'][aoi_id]
            if i == 0:
                timeseries = [(year, month, mask, s1, s2) for year, month, mask, s1, s2 in metadata if s1]
            else:
                timeseries = [(year, month, mask, s1, s2) for year, month, mask, s1, s2 in metadata if s1 and s2]
            f1_scores = []

            for j, ts in enumerate(timeseries):
                year, month, mask, *_ = ts
                if mask_helpers.is_fully_masked(dataset, aoi_id, year, month):
                    continue
                label = label_helpers.load_label(aoi_id, year, month)
                pred = input_helpers.load_prediction_raw(dataset, aoi_id, year, month, config_name)
                pred = pred > 0.5
                f1_scores.append(metrics.compute_f1_score(pred, label))
            data.append(f1_scores)

        center_pos = np.arange(len(aoi_ids))
        offset = (width + inbetween_space) / 2
        pos = center_pos - offset if i == 0 else center_pos + offset
        pl = ax.boxplot(data, positions=pos, sym='', widths=width)
        set_box_color(pl, colors[i])

        # draw temporary red and blue lines and use them to create a legend
        ax.plot([], c=colors[i], label=labels[i])

    if numeric_names:
        x_tick_labels = [f'AOI {i + 1}' for i in range(len(aoi_ids))]
    else:
        x_tick_labels = [aoi_id[4:15] for aoi_id in labels]

    ax.set_xticks(center_pos)
    ax.set_xticklabels(x_tick_labels, rotation=90, fontsize=fontsize)
    ax.set_ylabel('F1 score', fontsize=fontsize)
    y_ticks = np.linspace(0, 1, 6)
    ax.set_ylim((0, 1))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.1f}' for y_tick in y_ticks], fontsize=fontsize)
    ax.legend(ncol=2, handletextpad=0.4, columnspacing=1.2, frameon=False, loc='upper center', fontsize=fontsize)
    plt.show()


def all_sites_plot(dataset: str, save_plot: bool = False):
    pass


def change_detection_comparison():
    pass


if __name__ == '__main__':
    ds = 'spacenet7'
    timeseries_length_comparison(ds, numeric_names=True)
    config_names = ['sar_jaccardmorelikeloss', 'fusionda_cons05_jaccardmorelikeloss']
    # urban_extraction_comparison(config_names, ds, numeric_names=True)