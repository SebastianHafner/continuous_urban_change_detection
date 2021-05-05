from pathlib import Path
from utils import geofiles, visualization, dataset_helpers
import matplotlib.pyplot as plt
import numpy as np


def load_label_timeseries(aoi_id: str) -> np.ndarray:
    dates = dataset_helpers.get_timeseries('spacenet7_s1s2_dataset', aoi_id)
    buildings_path = dataset_helpers.dataset_path() / aoi_id / 'buildings'
    label_cube = None
    for i, (year, month, mask) in enumerate(dates):
        label_file = buildings_path / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
        label, _, _ = geofiles.read_tif(label_file, first_band_only=True)
        label = label > 0
        if label_cube is None:
            label_cube = np.zeros((*label.shape, len(dates)), dtype=np.uint8)
        label_cube[label, i] = 1
    return label_cube


def generate_change_date_label(aoi_id: str) -> np.ndarray:
    label_cube = load_label_timeseries(aoi_id)
    length_ts = dataset_helpers.length_timeseries('spacenet7_s1s2_dataset', aoi_id)

    change_date_label = np.zeros((label_cube.shape[0], label_cube.shape[1]), dtype=np.float32)
    for i in range(1, length_ts):
        change = np.logical_and(label_cube[:, :, i-1] == 0, label_cube[:, :, i] == 1)
        change_date_label[change] = i

    return change_date_label


def generate_change_label(dataset: str, aoi_id: str) -> np.ndarray:
    # computing it for spacenet7 (change between first and last label)
    if dataset == 'spacenet7_s1s2_dataset':
        label_cube = load_label_timeseries(aoi_id)
        length_ts = dataset_helpers.length_timeseries(dataset, aoi_id)
        sum_arr = np.sum(label_cube, axis=-1)
        change = np.logical_and(sum_arr != 0, sum_arr != length_ts)
    # for oscd the change label corresponds to the normal label
    else:
        label_file = dataset_helpers.root_path() / 'oscd_multitemporal_dataset' / aoi_id / f'change_{aoi_id}.tif'
        change, _, _ = geofiles.read_tif(label_file)
    return change.astype(np.uint8)


def load_label_in_timeseries(aoi_id: str, index: int, ignore_bad_data: bool = True) -> np.ndarray:
    dates = dataset_helpers.get_timeseries('spacenet7_s1s2_dataset', aoi_id, ignore_bad_data)
    buildings_path = dataset_helpers.dataset_path() / aoi_id / 'buildings'
    year, month, _ = dates[index]
    label_file = buildings_path / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
    label, _, _ = geofiles.read_tif(label_file)
    label = np.squeeze(label)
    return label


if __name__ == '__main__':
    pass
