from pathlib import Path
import yaml


def settings() -> dict:
    with open(str(Path.cwd() / 'settings.yaml')) as file:
        s = yaml.load(file, Loader=yaml.FullLoader)
    return s


SETTINGS = settings()


# dataset names
def spacenet7_dataset_name() -> str:
    return Path(SETTINGS['DATASET_NAMES']['SPACENET7'])


def oscd_dataset_name() -> str:
    return Path(SETTINGS['DATASET_NAMES']['OSCD'])


# dataset paths
def root_path() -> Path:
    return Path(SETTINGS['PATHS']['ROOT_PATH'])


# path to origin SpaceNet7 dataset
def spacenet7_path() -> Path:
    return Path(SETTINGS['PATHS']['SPACENET7_PATH'])


def oscd_path() -> Path:
    return Path(SETTINGS['PATHS']['OSCD_PATH'])


def config_name() -> str:
    sensor = SETTINGS['INPUT']['SENSOR']
    config_name_dict = SETTINGS['INPUT']['CONFIG_NAME_DICT']
    if sensor == 'sentinel1':
        return config_name_dict['sar']
    elif sensor == 'sentinel2':
        return config_name_dict['optical']
    else:
        return config_name_dict['fusion']


def include_masked() -> bool:
    return SETTINGS['INCLUDE_MASKED_DATA']


# plotting
def fontsize() -> int:
    return SETTINGS['PLOTTING']['FONTSIZE']


def plotsize() -> int:
    return SETTINGS['PLOTTING']['PLOTSIZE']


# sensor settings
def subset_activated(dataset: str) -> bool:
    if dataset == 'spacenet7':
        return SETTINGS['SUBSET_SPACENET7']['ACTIVATE']
    else:
        return SETTINGS['SUBSET_OSCD']['ACTIVATE']


def subset_aois(dataset: str) -> list:
    if dataset == 'spacenet7':
        return SETTINGS['SUBSET_SPACENET7']['AOI_IDS']
    else:
        return SETTINGS['SUBSET_OSCD']['AOI_IDS']


def input_sensor() -> str:
    return SETTINGS['INPUT']['SENSOR']


def input_type() -> str:
    return SETTINGS['INPUT']['TYPE']


def input_band() -> str:
    return SETTINGS['INPUT']['BAND']


def consistent_timeseries_length() -> bool:
    return SETTINGS['CONSISTENT_TIMESERIES_LENGTH']
