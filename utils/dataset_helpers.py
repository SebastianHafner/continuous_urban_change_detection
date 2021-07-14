from pathlib import Path
from utils import geofiles, config
import preprocess_spacenet7
import preprocess_oscd


def dataset_name(dataset: str) -> str:
    return config.spacenet7_dataset_name()if dataset == 'spacenet7' else config.oscd_dataset_name()


def bad_data(dataset: str) -> dict:
    bad_data_file = Path.cwd() / 'bad_data' / f'bad_data_{dataset_name(dataset)}.json'
    bad_data = geofiles.load_json(bad_data_file)
    return bad_data


def missing_aois() -> list:
    file = Path.cwd() / 'missing_aois.json'
    missing = geofiles.load_json(file)
    return missing


def spacenet7_timestamps() -> dict:
    timestamps_file = config.dataset_path('spacenet7') / 'spacenet7_timestamps.json'
    if not timestamps_file.exists():
        preprocess_spacenet7.assemble_spacenet7_timestamps()
    assert(timestamps_file.exists())
    timestamps = geofiles.load_json(timestamps_file)
    return timestamps


def oscd_timestamps() -> dict:
    timestamps_file = config.dataset_path('oscd') / 'oscd_timestamps.json'
    if not timestamps_file.exists():
        preprocess_oscd.assemble_oscd_timestamps()
    assert (timestamps_file.exists())
    timestamps = geofiles.load_json(timestamps_file)
    return timestamps


def timestamps(dataset: str) -> dict:
    return spacenet7_timestamps() if dataset == 'spacenet7' else oscd_timestamps()


# metadata functions
def oscd_metadata() -> dict:
    metadata_file = config.dataset_path('oscd') / 'metadata.json'
    if not metadata_file.exists():
        preprocess_oscd.generate_oscd_metadata_file()
    assert (metadata_file.exists())
    metadata = geofiles.load_json(metadata_file)
    return metadata


def spacenet7_metadata() -> dict:
    metadata_file = config.dataset_path('spacenet7') / 'metadata.json'
    if not metadata_file.exists():
        preprocess_spacenet7.generate_spacenet7_metadata_file()
    assert (metadata_file.exists())
    metadata = geofiles.load_json(metadata_file)
    return metadata


def metadata(dataset: str) -> dict:
    return spacenet7_metadata() if dataset == 'spacenet7' else oscd_metadata()


def aoi_metadata(dataset: str, aoi_id: str) -> list:
    md = metadata(dataset)
    return md['aois'][aoi_id]


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


def date2index(date: list) -> int:
    ref_value = 2019 * 12 + 1
    year, month = date
    return year * 12 + month - ref_value


# include masked data is only
def get_timeseries(dataset: str, aoi_id: str, include_masked_data: bool = False, ignore_bad_data: bool = True) -> list:
    aoi_md = aoi_metadata(dataset, aoi_id)

    # unpacking settings
    s = settings()
    input_sensor = s['INPUT']['SENSOR']
    consistent_length = s['CONSISTENT_TIMESERIES_LENGTH']

    if ignore_bad_data:
        if input_sensor == 'sentinel1':
            timeseries = [[y, m, mask, s1, s2] for y, m, mask, s1, s2 in aoi_md if s1]
        elif input_sensor == 'sentinel2':
            timeseries = [[y, m, mask, s1, s2] for y, m, mask, s1, s2 in aoi_md if s2]
        else:
            timeseries = [[y, m, mask, s1, s2] for y, m, mask, s1, s2 in aoi_md if s1 and s2]

        if consistent_length:
            # make sure that for the first and last timestamps all images (i.e., s1, s2 and planet) are clean
            clean_indices = [i for i, (_, _, mask, s1, s2) in enumerate(timeseries) if not mask and s1 and s2]
            min_clean, max_clean = min(clean_indices), max(clean_indices)
            timeseries = timeseries[min_clean:max_clean + 1]

        if include_masked_data:
            # trim time series at beginning and end such that it starts and ends with an unmasked timestamp
            unmasked_indices = [i for i, (_, _, mask, *_) in enumerate(timeseries) if not mask]
            min_unmasked, max_unmasked = min(unmasked_indices), max(unmasked_indices)
            timeseries = timeseries[min_unmasked:max_unmasked + 1]
        else:
            # remove all masked timestamps
            timeseries = [[y, m, mask, s1, s2] for y, m, mask, s1, s2 in aoi_md if not mask]
    else:
        timeseries = aoi_md
    return timeseries


def length_timeseries(dataset: str, aoi_id: str, include_masked_data: bool = False,
                      ignore_bad_data: bool = True) -> int:
    ts = get_timeseries(dataset, aoi_id, include_masked_data, ignore_bad_data)
    return len(ts)


def get_date_from_index(index: int, dataset: str, aoi_id: str, include_masked_data: bool = False,
                        ignore_bad_data: bool = True) -> int:
    ts = get_timeseries(dataset, aoi_id, include_masked_data, ignore_bad_data)
    year, month, *_ = ts[index]
    return year, month


def get_aoi_ids(dataset: str, exclude_missing: bool = True) -> list:
    ts = timestamps(dataset)
    s = settings()
    if dataset == 'spacenet7':
        if s['SUBSET_SPACENET7']['ACTIVATE']:
            aoi_ids = s['SUBSET_SPACENET7']['AOI_IDS']
        else:
            # handle exclude missing aoi ids
            aoi_ids = [aoi_id for aoi_id in ts.keys() if not (exclude_missing and aoi_id in missing_aois())]
    else:
        if s['SUBSET_OSCD']['ACTIVATE']:
            aoi_ids = s['SUBSET_OSCD']['AOI_IDS']
        else:
            aoi_ids = ts.keys()
    return sorted(aoi_ids)


def include_masked() -> bool:
    s = settings()
    return s['INCLUDE_MASKED_DATA']


def get_geo(dataset: str, aoi_id: str) -> tuple:
    folder = dataset_path(dataset) / aoi_id / 'sentinel1'
    file = [f for f in folder.glob('**/*') if f.is_file()][0]
    _, transform, crs = geofiles.read_tif(file)
    return transform, crs


def get_yx_size(dataset: str, aoi_id: str) -> tuple:
    md = metadata(dataset)
    return md['yx_sizes'][aoi_id]


def date2str(date: list):
    year, month, *_ = date
    return f'{year-2000:02d}-{month:02d}'

