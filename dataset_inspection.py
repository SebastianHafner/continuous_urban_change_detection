from pathlib import Path
from utils import geofiles, visualization
import matplotlib.pyplot as plt
import numpy as np


DATASET_PATH = Path('/storage/shafner/continuous_urban_change_detection/spacenet7_s1s2_dataset')


def get_time_series(aoi_id: str) -> list:
    metadata_file = DATASET_PATH / 'metadata.geojson'
    metadata = geofiles.load_json(metadata_file)
    return metadata['sites'][aoi_id]


def visualize_time_series(aoi_id: str):
    dates = get_time_series(aoi_id)
    n = len(dates)
    n_rows = 3
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    aoi_path = DATASET_PATH / aoi_id
    for i, (year, month) in enumerate(dates):
        s1_file = aoi_path / 'sentinel1' / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
        visualization.plot_sar(axs[0, i], s1_file)
        s2_file = aoi_path / 'sentinel2' / f'sentinel2_{aoi_id}_{year}_{month:02d}.tif'
        visualization.plot_optical(axs[1, i], s2_file)
        label_file = aoi_path / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
        visualization.plot_buildings(axs[2, i], label_file)
        axs[0, i].set_title(f'{year}-{month:02d}')

    plt.show()


def visualize_construction(aoi_id: str):
    dates = get_time_series(aoi_id)
    n = len(dates)
    n_rows = 2
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    aoi_path = DATASET_PATH / aoi_id
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
    # visualize_time_series('L15-0331E-1257N_1327_3160_13')
    visualize_construction('L15-0331E-1257N_1327_3160_13')