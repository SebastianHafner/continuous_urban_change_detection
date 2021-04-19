from pathlib import Path
from utils import geofiles


def root_path():
    return Path('/storage/shafner/continuous_urban_change_detection')


def dataset_path():
    return Path('/storage/shafner/continuous_urban_change_detection/spacenet7_s1s2_dataset')


def date2index(date: list) -> int:
    ref_value = 2019 * 12 + 1
    year, month = date
    return year * 12 + month - ref_value


def get_time_series(aoi_id: str, ignore_bad_data: bool = True) -> list:
    metadata_file = dataset_path() / 'metadata.json'
    metadata = geofiles.load_json(metadata_file)
    ts = metadata['sites'][aoi_id]
    if ignore_bad_data:
        bad_data_file = Path.cwd() / 'bad_data.json'
        bad_data = geofiles.load_json(bad_data_file)
        ts = [time_stamp for time_stamp in ts if not time_stamp in bad_data[aoi_id]]
    return ts


def get_all_ids() -> list:
    metadata_file = dataset_path() / 'metadata.json'
    metadata = geofiles.load_json(metadata_file)
    return metadata['sites'].keys()


def get_geo(aoi_id: str) -> tuple:
    dates = get_time_series(aoi_id, ignore_bad_data=False)
    year, month = dates[0]
    buildings_file = dataset_path() / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
    _, transform, crs = geofiles.read_tif(buildings_file)
    return transform, crs
