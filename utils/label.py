from pathlib import Path
from utils import geofiles, visualization, dataset_helpers
import matplotlib.pyplot as plt
import numpy as np


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


def generate_timeseries_label(aoi_id: str) -> np.ndarray:
    dates = dataset_helpers.get_time_series(aoi_id)
    buildings_path = dataset_helpers.dataset_path() / aoi_id / 'buildings'
    n = len(dates)
    assembled_label = None
    for i, (year, month) in enumerate(dates):
        label_file = buildings_path / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
        label, _, _ = geofiles.read_tif(label_file)
        label = label > 0
        if assembled_label is None:
            assembled_label = np.zeros((*label.shape, n), dtype=np.uint8)
        assembled_label[label, i] = 1

    return assembled_label


if __name__ == '__main__':
    label = generate_endtoend_label('L15-0331E-1257N_1327_3160_13')
    fig, ax = plt.subplots()
    visualization.plot_endtoend_label(ax, label)
    plt.show()