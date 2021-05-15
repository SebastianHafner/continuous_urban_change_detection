from pathlib import Path
from utils import geofiles, visualization, dataset_helpers, prediction_helpers, label_helpers, metrics
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


def visualize_all_data(dataset: str, aoi_id: str, include_f1_score: bool = False,
                       save_plot: bool = False):

    if aoi_id in dataset_helpers.missing_aois():
        return

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
    aoi_ids = dataset_helpers.get_all_ids(dataset)
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


def sanity_check_change_detection_label(dataset: str, aoi_id: str, include_masked_data: bool = False,
                                        include_buildings_label: bool = True, save_plot: bool = False):
    # buildings labels only exist for spacenet7 dataset
    include_buildings_label = include_buildings_label and dataset == 'spacenet7'
    dates = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)

    n_plots = 5 if include_buildings_label else 3
    fig, axs = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    # first
    year_first, month_first, *_ = dates[0]
    visualization.plot_optical(axs[0], dataset, aoi_id, year_first, month_first)
    if include_buildings_label:
        visualization.plot_buildings(axs[1], aoi_id, year_first, month_first)

    # last
    year_last, month_last, *_ = dates[-1]
    visualization.plot_optical(axs[2 if include_buildings_label else 1], dataset, aoi_id, year_last, month_last)
    if include_buildings_label:
        visualization.plot_buildings(axs[3], aoi_id, year_last, month_last)

    # change label
    visualization.plot_change_label(axs[4 if include_buildings_label else 2], dataset, aoi_id, include_masked_data)

    if not save_plot:
        plt.show()
    else:
        dataset_name = dataset_helpers.dataset_name(dataset)
        output_path = dataset_helpers.root_path() / 'plots' / 'sanity_check' / 'change_label' / dataset_name
        output_path.mkdir(exist_ok=True)
        output_file = output_path / f'{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def sanity_check_change_dating_label(dataset: str, aoi_id: str, include_masked_data: bool = False,
                                     save_plot: bool = False):
    ts = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)
    n = len(ts)
    n_rows, n_cols = 1, n + 1
    plot_size = 3

    cmap = visualization.DateColorMap(n).get_cmap()

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * plot_size, n_rows * plot_size))

    for i, (year, month, *_) in enumerate(tqdm(ts)):
        visualization.plot_optical(axs[i], dataset, aoi_id, year, month)
        title = f'{year}-{month:02d}'
        color = cmap(i)
        axs[i].set_title(title, c=color, fontsize=16, fontweight='bold')

    visualization.plot_change_date_label(axs[-1], aoi_id, include_masked_data)

    if not save_plot:
        plt.show()
    else:
        output_path = dataset_helpers.root_path() / 'plots' / 'sanity_check' / 'date_label'
        output_path.parent.mkdir(exist_ok=True)
        output_file = output_path / f'{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    ds = 'spacenet7'
    for aoi_id in dataset_helpers.get_aoi_ids(ds):
        # visualize_satellite_data(ds, aoi_id, save_plot=True)
        # visualize_all_data(ds, aoi_id, config_name=cfg, save_plot=True)
        # visualize_timeseries(ds, aoi_id, config_name=cfg, save_plot=True)
        # sanity_check_change_detection_label(ds, aoi_id, include_masked_data=True, save_plot=False)
        sanity_check_change_dating_label(ds, aoi_id, include_masked_data=True, save_plot=True)
        pass

    # sanity_check_change_detection_label(ds, 'L15-0331E-1257N_1327_3160_13', include_masked_data=False, save_plot=False)
    sanity_check_change_dating_label(ds, 'L15-1479E-1101N_5916_3785_13', include_masked_data=True, save_plot=False)
    # visualize_timeseries_length(ds)