from pathlib import Path
from utils import geofiles, visualization, dataset_helpers
import matplotlib.pyplot as plt
import numpy as np


def visualize_time_series(aoi_id: str, save_plot: bool = False):
    dates = dataset_helpers.get_time_series(aoi_id)
    n = len(dates)
    n_rows = 3
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    aoi_path = dataset_helpers.dataset_path() / aoi_id
    for i, (year, month) in enumerate(dates):
        s1_file = aoi_path / 'sentinel1' / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
        visualization.plot_sar(axs[0, i], s1_file)
        s2_file = aoi_path / 'sentinel2' / f'sentinel2_{aoi_id}_{year}_{month:02d}.tif'
        visualization.plot_optical(axs[1, i], s2_file)
        label_file = aoi_path / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
        visualization.plot_buildings(axs[2, i], label_file)
        axs[0, i].set_title(f'{year}-{month:02d}')

    if not save_plot:
        plt.show()
        plt.close(fig)
    else:
        output_file = dataset_helpers.root_path() / 'plots' / 'inspection' / f'time_series_{aoi_id}.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')


def visualize_construction(aoi_id: str):
    dates = dataset_helpers.get_time_series(aoi_id)
    n = len(dates)
    n_rows = 2
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    aoi_path = dataset_helpers.dataset_path() / aoi_id
    previous_label_file = None
    for i, (year, month) in enumerate(dates):
        s2_file = aoi_path / 'sentinel2' / f'sentinel2_{aoi_id}_{year}_{month:02d}.tif'
        visualization.plot_optical(axs[0, i], s2_file)
        label_file = aoi_path / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
        if i == 0:
            visualization.plot_buildings(axs[1, i], label_file)
        else:
            label, _, _ = geofiles.read_tif(label_file)
            previous_label, _, _ = geofiles.read_tif(previous_label_file)
            label = label > 0
            previous_label = previous_label > 0
            no_change = label == previous_label
            change = np.logical_not(no_change)
            visualization.plot_blackwhite(axs[1, i], change)
        previous_label_file = label_file
    plt.show()


if __name__ == '__main__':
    for aoi_id in dataset_helpers.get_all_ids():
        visualize_time_series(aoi_id, save_plot=True)
    # visualize_construction('L15-0331E-1257N_1327_3160_13')