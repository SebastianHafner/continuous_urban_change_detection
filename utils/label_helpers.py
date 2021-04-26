from pathlib import Path
from utils import geofiles, visualization, dataset_helpers
import matplotlib.pyplot as plt
import numpy as np


def generate_timeseries_label(aoi_id: str) -> np.ndarray:
    dates = dataset_helpers.get_time_series(aoi_id)
    buildings_path = dataset_helpers.dataset_path() / aoi_id / 'buildings'
    label_cube = None
    for i, (year, month) in enumerate(dates):
        label_file = buildings_path / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
        label, _, _ = geofiles.read_tif(label_file, first_band_only=True)
        label = label > 0
        if label_cube is None:
            label_cube = np.zeros((*label.shape, len(dates)), dtype=np.uint8)
        label_cube[label, i] = 1
    return label_cube

def generate_endtoend_label(aoi_id: str) -> np.ndarray:
    dates = dataset_helpers.get_time_series(aoi_id)
    buildings_path = dataset_helpers.dataset_path() / aoi_id / 'buildings'
    endtoend_label = None
    for i, (year, month) in enumerate(dates):
        label_file = buildings_path / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
        label, _, _ = geofiles.read_tif(label_file)
        label = label > 0
        if endtoend_label is None:
            endtoend_label = np.zeros(label.shape, dtype=np.uint8)
            endtoend_label[label] = 1
        else:
            current_builtup = endtoend_label > 0
            new_builtup = np.logical_and(np.logical_not(current_builtup), label)
            endtoend_label[new_builtup] = i + 1

    return endtoend_label


def generate_change_date_label(aoi_id: str) -> np.ndarray:
    label_cube = generate_timeseries_label(aoi_id)
    length_ts = dataset_helpers.length_time_series(aoi_id)

    change_date_label = np.zeros((label_cube.shape[0], label_cube.shape[1]), dtype=np.float32)
    for i in range(1, length_ts):
        change = np.logical_and(label_cube[:, :, i-1] == 0, label_cube[:, :, i] == 1)
        change_date_label[change] = i

    return change_date_label


def generate_change_label(aoi_id: str) -> np.ndarray:
    label_cube = generate_timeseries_label(aoi_id)
    length_ts = dataset_helpers.length_time_series(aoi_id)

    sum_arr = np.sum(label_cube, axis=-1)
    change = np.logical_and(sum_arr != 0, sum_arr != length_ts)
    return change.astype(np.uint8)


def load_endtoend_label(aoi_id: str) -> np.ndarray:
    label_file = dataset_helpers.dataset_path() / aoi_id / f'label_endtoend_{aoi_id}.tif'
    label, _, _ = geofiles.read_tif(label_file)
    return label


def get_label_in_timeseries(aoi_id: str, index: int, ignore_bad_data: bool = True) -> np.ndarray:
    dates = dataset_helpers.get_time_series(aoi_id, ignore_bad_data)
    buildings_path = dataset_helpers.dataset_path() / aoi_id / 'buildings'
    year, month = dates[index]
    label_file = buildings_path / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
    label, _, _ = geofiles.read_tif(label_file)
    label = np.squeeze(label)
    return label


if __name__ == '__main__':
    label = generate_endtoend_label('L15-0331E-1257N_1327_3160_13')
    fig, ax = plt.subplots()
    visualization.plot_endtoend_label(ax, label)
    plt.show()