from pathlib import Path
from utils import geofiles, visualization, dataset_helpers
import matplotlib.pyplot as plt
import numpy as np


def generate_timeseries_prediction(config_name: str, aoi_id: str) -> np.ndarray:
    dates = dataset_helpers.get_time_series(aoi_id)
    predictions_path = dataset_helpers.dataset_path() / aoi_id / config_name
    n = len(dates)
    assembled_label = None
    for i, (year, month) in enumerate(dates):
        label_file = predictions_path / f'pred_{aoi_id}_{year}_{month:02d}.tif'
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