from pathlib import Path
from utils import geofiles


def root_path():
    return Path('/storage/shafner/continuous_urban_change_detection')


def dataset_path():
    return Path('/storage/shafner/continuous_urban_change_detection/spacenet7_s1s2_dataset')


def load_aoi_selection():
    file = Path.cwd() / 'aoi_selection.json'
    selection = geofiles.load_json(file)
    return selection


def date2index(date: list) -> int:
    ref_value = 2019 * 12 + 1
    year, month = date
    return year * 12 + month - ref_value


def get_time_series(dataset: str, aoi_id: str, ignore_bad_data: bool = True) -> list:
    metadata_file = root_path() / dataset / 'metadata.json'
    metadata = geofiles.load_json(metadata_file)
    time_series = metadata['sites'][aoi_id]
    if ignore_bad_data:
        bad_data_file = Path.cwd() / f'bad_data_{dataset}.json'
        bad_data = geofiles.load_json(bad_data_file)
        time_series = [ts for ts in time_series if not [ts[0], ts[1]] in bad_data[aoi_id]]
    return time_series


def length_time_series(dataset: str, aoi_id: str, ignore_bad_data: bool = True) -> int:
    ts = get_time_series(dataset, aoi_id, ignore_bad_data=ignore_bad_data)
    return len(ts)


def get_all_ids(dataset: str) -> list:
    metadata_file = root_path() / dataset / 'metadata.json'
    metadata = geofiles.load_json(metadata_file)
    return metadata['sites'].keys()


def get_geo(dataset: str, aoi_id: str) -> tuple:
    dates = get_time_series(dataset, aoi_id, ignore_bad_data=False)
    year, month = dates[0]
    buildings_file = root_path() / dataset / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
    _, transform, crs = geofiles.read_tif(buildings_file)
    return transform, crs


def date2str(date: list):
    year, month = date
    return f'{year-2000:02d}-{month:02d}'
