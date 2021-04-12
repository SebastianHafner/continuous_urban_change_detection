from pathlib import Path
from utils import geofiles, visualization
import matplotlib.pyplot as plt
import numpy as np


DATASET_PATH = Path('/storage/shafner/continuous_urban_change_detection/spacenet7_s1s2_dataset')


def get_time_series(aoi_id: str) -> list:
    metadata_file = DATASET_PATH / 'metadata.geojson'
    metadata = geofiles.load_json(metadata_file)
    return metadata['sites'][aoi_id]


def generate_endtoend_label(aoi_id) -> np.ndarray:
    dates = get_time_series(aoi_id)
    endtoend_label = None
    for i, (year, month) in enumerate(dates):
        label_file = DATASET_PATH / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
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


if __name__ == '__main__':
    label = generate_endtoend_label('L15-0331E-1257N_1327_3160_13')
    fig, ax = plt.subplots()
    visualization.plot_endtoend_label(ax, label)
    plt.show()