from pathlib import Path
from utils import geofiles
import numpy as np
import preprocess_spacenet7
import preprocess_oscd

# GLOBAL VARIABLES
ROOT_PATH = '/storage/shafner/continuous_urban_change_detection'
SPACENET7_PATH = '/storage/shafner/spacenet7'  # this is the origin SpaceNet7 dataset
SPACENET7_DATASET_NAME = 'spacenet7_s1s2_dataset_v2'
OSCD_DATASET_NAME = 'oscd_s1s2_dataset'
CONFIG_NAME = 'fusionda_cons05_jaccardmorelikeloss'


# dataset names
def spacenet7_dataset_name() -> str:
    return SPACENET7_DATASET_NAME


def oscd_dataset_name() -> str:
    return OSCD_DATASET_NAME


def dataset_name(dataset: str) -> str:
    return spacenet7_dataset_name()if dataset == 'spacenet7' else oscd_dataset_name()


# dataset paths
def root_path() -> Path:
    return Path(ROOT_PATH)


def dataset_path(dataset: str) -> Path:
    return root_path() / dataset_name(dataset)


# path to origin SpaceNet7 dataset
def spacenet7_path() -> Path:
    return Path(SPACENET7_PATH)


def bad_data(dataset: str) -> dict:
    bad_data_file = Path.cwd() / 'bad_data' / f'bad_data_{dataset_name(dataset)}.json'
    bad_data = geofiles.load_json(bad_data_file)
    return bad_data


def missing_aois() -> list:
    file = Path.cwd() / 'missing_aois.json'
    missing = geofiles.load_json(file)
    return missing


def spacenet7_timestamps() -> dict:
    timestamps_file = dataset_path('spacenet7') / 'spacenet7_timestamps.json'
    if not timestamps_file.exists():
        preprocess_spacenet7.assemble_spacenet7_timestamps()
    assert(timestamps_file.exists())
    timestamps = geofiles.load_json(timestamps_file)
    return timestamps


def oscd_timestamps() -> dict:
    timestamps_file = dataset_path('oscd') / 'oscd_timestamps.json'
    if not timestamps_file.exists():
        preprocess_oscd.assemble_oscd_timestamps()
    assert (timestamps_file.exists())
    timestamps = geofiles.load_json(timestamps_file)
    return timestamps


def timestamps(dataset: str) -> dict:
    return spacenet7_timestamps() if dataset == 'spacenet7' else oscd_timestamps()


# metadata functions
# TODO: implement this
def oscd_metadata() -> dict:
    pass


def spacenet7_metadata() -> dict:
    metadata_file = dataset_path('spacenet7') / 'metadata.json'
    if not metadata_file.exists():
        preprocess_spacenet7.generate_spacenet7_metadata_file()
    assert (metadata_file.exists())
    metadata = geofiles.load_json(metadata_file)
    return metadata


def metadata(dataset: str) -> dict:
    return spacenet7_metadata() if dataset == 'spacenet7' else oscd_metadata()


def metadata_index(dataset: str, aoi_id: str, year: int, month: int) -> int:
    md = metadata(dataset)[aoi_id]
    for i, (y, m, *_) in enumerate(md):
        if y == year and month == month:
            return i


def metadata_timestamp(dataset: str, aoi_id: str, year: int, month: int) -> int:
    md = metadata(dataset)[aoi_id]
    for i, ts in enumerate(md):
        y, m, *_ = ts
        if y == year and month == month:
            return ts


def config_name() -> str:
    return CONFIG_NAME


def date2index(date: list) -> int:
    ref_value = 2019 * 12 + 1
    year, month = date
    return year * 12 + month - ref_value


# include masked data is only
def get_timeseries(dataset: str, aoi_id: str, include_masked_data: bool = False, ignore_bad_data: bool = True) -> list:
    timeseries = metadata(dataset)['aois'][aoi_id]
    if ignore_bad_data:
        if include_masked_data:
            timeseries = [[y, m, mask, s1, s2] for y, m, mask, s1, s2 in timeseries if s1 and s2]
            # trim time series at beginning and end such that it starts and ends with an unmasked timestamp
            unmasked_indices = [i for i, (_, _, mask, *_) in enumerate(timeseries) if not mask]
            min_unmasked, max_unmasked = min(unmasked_indices), max(unmasked_indices)
            timeseries = timeseries[min_unmasked:max_unmasked + 1]
        else:
            timeseries = [[y, m, mask, s1, s2] for y, m, mask, s1, s2 in timeseries if not mask and (s1 and s2)]
    return timeseries


def length_timeseries(dataset: str, aoi_id: str, include_masked_data: bool = False,
                      ignore_bad_data: bool = True) -> int:
    ts = get_timeseries(dataset, aoi_id, include_masked_data, ignore_bad_data)
    return len(ts)


def get_aoi_ids(dataset: str, exclude_missing: bool = True) -> list:
    ts = timestamps(dataset)
    if dataset == 'spacenet7':
        aoi_ids = [aoi_id for aoi_id in ts.keys() if not (exclude_missing and aoi_id in missing_aois())]
    else:
        aoi_ids = ts.keys()
    return sorted(aoi_ids)


# TODO: make this alos work for OSCD dataset
def get_geo(dataset: str, aoi_id: str) -> tuple:
    dates = get_timeseries(dataset, aoi_id, ignore_bad_data=False)
    year, month = dates[0]
    buildings_file = dataset_path(dataset) / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
    _, transform, crs = geofiles.read_tif(buildings_file)
    return transform, crs


# TODO: would make sense to cache xy size
def get_yx_size(dataset: str, aoi_id: str) -> tuple:
    folder = dataset_path(dataset) / aoi_id / 'sentinel1'
    file = [f for f in folder.glob('**/*') if f.is_file()][0]
    arr, transform, crs = geofiles.read_tif(file)
    return arr.shape[0], arr.shape[1]


def date2str(date: list):
    year, month, *_ = date
    return f'{year-2000:02d}-{month:02d}'
