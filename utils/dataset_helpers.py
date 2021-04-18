from pathlib import Path
from utils import geofiles


def root_path():
    return Path('/storage/shafner/continuous_urban_change_detection')


def dataset_path():
    return Path('/storage/shafner/continuous_urban_change_detection/spacenet7_s1s2_dataset')


def get_time_series(aoi_id: str) -> list:
    metadata_file = dataset_path() / 'metadata.json'
    metadata = geofiles.load_json(metadata_file)
    return metadata['sites'][aoi_id]


def get_all_ids() -> list:
    metadata_file = dataset_path() / 'metadata.json'
    metadata = geofiles.load_json(metadata_file)
    return metadata['sites'].keys()
