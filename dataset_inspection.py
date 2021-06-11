from pathlib import Path
from utils import geofiles, visualization, dataset_helpers, prediction_helpers, label_helpers, metrics, mask_helpers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def visualize_satellite_data(dataset: str, aoi_id: str, save_plot: bool = False):
    timestamps = dataset_helpers.timestamps(dataset)
    ts = timestamps[aoi_id]
    n = len(ts)
    n_rows = 4
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    for i, (year, month, _) in enumerate(tqdm(ts)):
        visualization.plot_optical(axs[0, i], dataset, aoi_id, year, month, vis='true_color')
        visualization.plot_optical(axs[1, i], dataset, aoi_id, year, month, vis='false_color')
        visualization.plot_sar(axs[2, i], dataset, aoi_id, year, month, vis='VV')
        visualization.plot_sar(axs[3, i], dataset, aoi_id, year, month, vis='VH')

        title = f'{i} {year}-{month:02d}'

        axs[0, i].set_title(title, c='k', fontsize=16, fontweight='bold')

    if not save_plot:
        plt.show()
    else:
        dataset_name = dataset_helpers.dataset_name(dataset)
        output_file = dataset_helpers.root_path() / 'plots' / 'inspection' / dataset_name / f'{aoi_id}_satellite.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_all_data(dataset: str, aoi_id: str, save_plot: bool = False):

    metadata = dataset_helpers.metadata(dataset)
    ts = metadata['aois'][aoi_id]
    n = len(ts)
    gt_available = True if dataset == 'spacenet7' else False
    n_rows = 4 if gt_available else 3
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    for i, (year, month, mask, s1, s2) in enumerate(tqdm(ts)):
        visualization.plot_sar(axs[0, i], ds, aoi_id, year, month)
        visualization.plot_optical(axs[1, i], ds, aoi_id, year, month)

        if gt_available:
            visualization.plot_buildings(axs[2, i], aoi_id, year, month)

        # title
        title = f'{i} {year}-{month:02d}'
        if mask:
            color = 'blue'
        else:
            if s1 and s2:
                color = 'green'
            elif s1:
                color = 'orange'
            elif s2:
                color = 'cyan'
            else:
                color = 'red'
        if gt_available:
            y_true = label_helpers.load_label(aoi_id, year, month)
            y_pred = prediction_helpers.load_prediction(dataset, aoi_id, year, month)
            y_pred = y_pred > 0.5
            if mask_helpers.is_fully_masked(dataset, aoi_id, year, month):
                f1 = 'NaN'
            else:
                f1 = metrics.compute_f1_score(y_pred.flatten(), y_true.flatten())
                f1 = f'{f1*100:.1f}'
            title = f'{title} {f1}'

        axs[0, i].set_title(title, c=color, fontsize=12, fontweight='bold')
        visualization.plot_prediction(axs[n_rows - 1, i], dataset, aoi_id, year, month)

    if not save_plot:
        plt.show()
    else:
        dataset_name = dataset_helpers.dataset_name(dataset)
        output_file = dataset_helpers.root_path() / 'plots' / 'inspection' / dataset_name / f'{aoi_id}_data.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_timeseries_length(dataset: str):
    data, labels = [], []
    aoi_ids = dataset_helpers.get_aoi_ids(dataset)
    for aoi_id in tqdm(aoi_ids):
        if aoi_id in dataset_helpers.missing_aois():
            continue
        metadata = dataset_helpers.metadata(dataset)['aois'][aoi_id]
        n_clear = len([_ for _, _, mask, s1, s2 in metadata if not mask and (s1 and s2)])
        n_clear_masked = len([_ for _, _, mask, s1, s2 in metadata if mask and (s1 and s2)])
        data.append([n_clear, n_clear_masked])
        labels.append(aoi_id)

    data = sorted(data, key=lambda d: d[0])
    clear = [d[0] for d in data]
    clear_masked = [d[1] for d in data]
    labels = [aoi_id[4:15] for aoi_id in labels]
    print(data, labels)
    width = 0.5  # the width of the bars: can also be len(x) sequence

    fontsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(len(labels) / 2, 10))

    ax.bar(labels, clear, width, label='Clear')
    ax.bar(labels, clear_masked, width, bottom=clear, label='Clear (masked)')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_xlim((-0.5, len(labels) + 0.5))

    max_value = (max([c + cm for c, cm in data]) // 5 + 1) * 5
    y_ticks = np.arange(0, max_value + 1, 5)
    ax.set_ylim((0, max_value))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.0f}' for y_tick in y_ticks], fontsize=fontsize)
    ax.set_ylabel('n Timestamps', fontsize=fontsize)

    ax.legend(loc='upper left', fontsize=fontsize)

    plt.show()


if __name__ == '__main__':
    ds = 'spacenet7'
    for i, aoi_id in enumerate(dataset_helpers.get_aoi_ids(ds)):
        # visualize_satellite_data(ds, aoi_id, save_plot=True)
        visualize_all_data(ds, aoi_id, save_plot=True)
        # visualize_timeseries(ds, aoi_id, config_name=cfg, save_plot=True)
        pass

    # visualize_timeseries_length(ds)