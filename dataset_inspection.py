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


def visualize_time_series(dataset: str, aoi_id: str, config_name: str = None, include_f1_score: bool = False,
                          save_plot: bool = False):
    ts_complete = dataset_helpers.get_time_series(dataset, aoi_id, ignore_bad_data=False)
    ts_clear = dataset_helpers.get_time_series(dataset, aoi_id, ignore_bad_data=True)
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
        if include_f1_score and gt_available:
            label = label_helpers.get_label_in_timeseries(aoi_id, i, ignore_bad_data=False) > 0
            pred = prediction_helpers.get_prediction_in_timeseries(config_name, aoi_id, i, ignore_bad_data=False) > 0.5
            f1_score = metrics.compute_f1_score(pred, label)
            title += f' (F1 {f1_score:.2f})'

        if [year, month, mask] in ts_clear:
            color = 'green' if not mask else 'blue'
        else:
            color = 'orange' if not mask else 'red'

        axs[0, i].set_title(title, c=color, fontsize=16, fontweight='bold')

        if config_name is not None:
            visualization.plot_prediction(axs[n_rows - 1, i], dataset, config_name, aoi_id, year, month)

    if not save_plot:
        plt.show()
    else:
        output_file = dataset_helpers.root_path() / 'plots' / 'inspection' / dataset / f'time_series_{aoi_id}.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_first_and_last_optical(aoi_id: str, save_plot: bool = False):
    dates = dataset_helpers.get_time_series(aoi_id)
    fig, axs = plt.subplots(2, 1, figsize=(10, 20))

    # first
    year_first, month_first = dates[0]
    visualization.plot_optical(axs[0], aoi_id, year_first, month_first)

    # last
    year_last, month_last = dates[-1]
    visualization.plot_optical(axs[1], aoi_id, year_last, month_last)

    if not save_plot:
        plt.show()
        plt.close(fig)
    else:
        output_file = dataset_helpers.root_path() / 'plots' / 'first_last' / f'{aoi_id}.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # for i, aoi_id in enumerate(dataset_helpers.get_all_ids()):
    #     print(f'{i}: {aoi_id}')
    #     # visualize_first_and_last_optical(aoi_id, save_plot=True)
    #     visualize_time_series(aoi_id, config_name='fusionda_cons05_jaccardmorelikeloss',
    #                           include_f1_score=True, save_plot=True)
    #     pass
    ds = 'oscd_multitemporal_dataset'
    cfg = 'fusionda_cons05_jaccardmorelikeloss'
    for aoi_id in dataset_helpers.get_all_ids(ds):
        # visualize_satellite_data('oscd_multitemporal_dataset', aoi_id, save_plot=True)
        visualize_time_series(ds, aoi_id, config_name=cfg, save_plot=True)