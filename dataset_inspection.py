from pathlib import Path
from utils import geofiles, visualization, dataset_helpers, prediction_helpers, label_helpers, metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def visualize_satellite_data(dataset: str, aoi_id: str, save_plot: bool = False):
    ts_complete = dataset_helpers.get_time_series(dataset, aoi_id, ignore_bad_data=False)
    ts_clear = dataset_helpers.get_time_series(dataset, aoi_id, ignore_bad_data=True)
    n = len(ts_complete)
    n_rows = 2
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    for i, (year, month, mask) in enumerate(tqdm(ts_complete)):
        visualization.plot_sar(axs[0, i], dataset, aoi_id, year, month)
        visualization.plot_optical(axs[1, i], dataset, aoi_id, year, month)

        title = f'{year}-{month:02d}'

        if [year, month, mask] in ts_clear:
            color = 'green' if not mask else 'blue'
        else:
            color = 'orange' if not mask else 'red'

        axs[0, i].set_title(title, c=color, fontsize=16, fontweight='bold')

    if not save_plot:
        plt.show()
    else:
        output_file = dataset_helpers.root_path() / 'plots' / 'inspection' / dataset / f'satellite_data_{aoi_id}.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_timeseries(dataset: str, aoi_id: str, config_name: str = None, include_f1_score: bool = False,
                          save_plot: bool = False):
    ts_complete = dataset_helpers.get_timeseries(dataset, aoi_id, ignore_bad_data=False)
    ts_clear = dataset_helpers.get_timeseries(dataset, aoi_id, ignore_bad_data=True)
    n = len(ts_complete)
    gt_available = True if dataset == 'spacenet7_s1s2_dataset' else False
    n_rows = 3 if gt_available else 2
    if config_name is not None:
        n_rows += 1
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    for i, (year, month, mask) in enumerate(tqdm(ts_complete)):
        visualization.plot_sar(axs[0, i], ds, aoi_id, year, month)
        visualization.plot_optical(axs[1, i], ds, aoi_id, year, month)

        if gt_available:
            visualization.plot_buildings(axs[2, i], aoi_id, year, month)

        # title
        title = f'{year}-{month:02d}'
        # TODO: readd differentiation for masked gt
        color = 'green' if [year, month, mask] in ts_clear else 'red'
        axs[0, i].set_title(title, c=color, fontsize=16, fontweight='bold')

        if config_name is not None:
            visualization.plot_prediction(axs[n_rows - 1, i], config_name, dataset, aoi_id, year, month)

    if not save_plot:
        plt.show()
    else:
        output_file = dataset_helpers.root_path() / 'plots' / 'inspection' / dataset / f'time_series_{aoi_id}.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def sanity_check_change_detection_label(dataset: str, aoi_id: str, save_plot: bool = False):
    dates = dataset_helpers.get_timeseries(dataset, aoi_id)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # first
    year_first, month_first = dates[0][:-1]
    visualization.plot_optical(axs[0], dataset, aoi_id, year_first, month_first)

    # last
    year_last, month_last = dates[-1][:-1]
    visualization.plot_optical(axs[1], dataset, aoi_id, year_last, month_last)

    # change label
    visualization.plot_change_label(axs[2], dataset, aoi_id)

    if not save_plot:
        plt.show()
    else:
        output_path = dataset_helpers.root_path() / 'plots' / 'sanity_check' / 'change_label' / dataset
        output_path.mkdir(exist_ok=True)
        output_file = output_path / f'{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def sanity_check_change_dating_label(aoi_id: str, save_plot: bool = False):
    ts = dataset_helpers.get_time_series('spacenet7_s1s2_dataset', aoi_id)
    n = len(ts)
    n_rows, n_cols = 1, n + 1
    plot_size = 3

    cmap = visualization.DateColorMap(n).get_cmap()

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * plot_size, n_rows * plot_size))

    for i, (year, month, _) in enumerate(tqdm(ts)):
        visualization.plot_optical(axs[i], 'spacenet7_s1s2_dataset', aoi_id, year, month)
        title = f'{year}-{month:02d}'
        color = cmap(i)
        axs[i].set_title(title, c=color, fontsize=16, fontweight='bold')

    visualization.plot_change_date_label(axs[-1], aoi_id)

    if not save_plot:
        plt.show()
    else:
        output_path = dataset_helpers.root_path() / 'plots' / 'sanity_check' / 'date_label'
        output_path.parent.mkdir(exist_ok=True)
        output_file = output_path / f'{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    ds = 'oscd_multitemporal_dataset'
    ds = 'spacenet7_s1s2_dataset'
    cfg = 'fusionda_cons05_jaccardmorelikeloss'
    for aoi_id in dataset_helpers.get_all_ids(ds):
        # visualize_satellite_data('oscd_multitemporal_dataset', aoi_id, save_plot=True)

        # visualize_time_series(ds, aoi_id, config_name=cfg, save_plot=True)
        # sanity_check_change_detection_label(ds, aoi_id, save_plot=False)
        # sanity_check_change_dating_label(aoi_id, save_plot=True)
        pass

    sanity_check_change_detection_label(ds, 'L15-0358E-1220N_1433_3310_13', save_plot=False)