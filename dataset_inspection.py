from pathlib import Path
from utils import geofiles, visualization, dataset_helpers, prediction_helpers, label_helpers, metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def visualize_time_series(aoi_id: str, config_name: str = None, include_f1_score: bool = False,
                          save_plot: bool = False):
    all_dates = dataset_helpers.get_time_series(aoi_id, ignore_bad_data=False)
    clear_dates = dataset_helpers.get_time_series(aoi_id, ignore_bad_data=True)
    n = len(all_dates)
    n_rows = 3 if config_name is None else 4
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    for i, (year, month) in enumerate(tqdm(all_dates)):
        visualization.plot_sar(axs[0, i], aoi_id, year, month)
        visualization.plot_optical(axs[1, i], aoi_id, year, month)
        visualization.plot_buildings(axs[2, i], aoi_id, year, month)

        # title
        title = f'{year}-{month:02d}'
        if include_f1_score:
            label = label_helpers.get_label_in_timeseries(aoi_id, i, ignore_bad_data=False) > 0
            pred = prediction_helpers.get_prediction_in_timeseries(config_name, aoi_id, i, ignore_bad_data=False) > 0.5
            f1_score = metrics.compute_f1_score(pred, label)
            title += f' (F1 {f1_score:.2f})'

        color = 'k' if [year, month] in clear_dates else 'red'
        axs[0, i].set_title(title, c=color, fontsize=18, fontweight='bold')

        if config_name is not None:
            visualization.plot_prediction(axs[3, i], config_name, aoi_id, year, month)



    if not save_plot:
        plt.show()
    else:
        output_file = dataset_helpers.root_path() / 'plots' / 'inspection' / f'time_series_{aoi_id}.png'
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
    for aoi_id in dataset_helpers.get_all_ids():
        print(aoi_id)
        # visualize_first_and_last_optical(aoi_id, save_plot=True)
        visualize_time_series(aoi_id, config_name='fusionda_cons05_jaccardmorelikeloss',
                              include_f1_score=True, save_plot=True)
    # visualize_construction('L15-0331E-1257N_1327_3160_13')