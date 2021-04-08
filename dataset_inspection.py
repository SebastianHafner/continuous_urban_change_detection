from pathlib import Path
from utils import geofiles
import matplotlib.pyplot as plt


DATASET_PATH = Path('/storage/shafner/continuous_urban_change_detection/spacenet7_s1s2_dataset')


def visualize_time_series(aoi_id: str):
    metadata_file = DATASET_PATH / 'metadata.geojson'
    metadata = geofiles.load_json(metadata_file)
    site_metadata = [site for site in metadata['sites'] if site['name'] == aoi_id][0]
    dates = site_metadata['dates']
    n = len(dates)
    n_rows = 3
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    aoi_path = DATASET_PATH / aoi_id
    for i, (year, month) in enumerate(dates):
        s1_file = aoi_path / 'sentinel1' / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
        s2_file = aoi_path / 'sentinel2' / f'sentinel2_{aoi_id}_{year}_{month:02d}.tif'
        label_file = aoi_path / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'


if __name__ == '__main__':
    visualize_time_series('L15-0331E-1257N_1327_3160_13')
