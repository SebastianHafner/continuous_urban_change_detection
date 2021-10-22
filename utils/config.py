from pathlib import Path
import yaml


def settings() -> dict:
    with open(str(Path.cwd() / 'settings.yaml')) as file:
        s = yaml.load(file, Loader=yaml.FullLoader)
    return s


SETTINGS = settings()


def dataset_name() -> str:
    return Path(SETTINGS['DATASET_NAME'])


# dataset paths
def root_path() -> Path:
    return Path(SETTINGS['PATHS']['ROOT_PATH'])


# path to origin SpaceNet7 dataset
def spacenet7_path() -> Path:
    return Path(SETTINGS['PATHS']['SPACENET7_PATH'])


def config_name() -> str:
    sensor = SETTINGS['INPUT']['SENSOR']
    config_name_dict = SETTINGS['INPUT']['CONFIG_NAME_DICT']
    if sensor == 'sentinel1':
        return config_name_dict['sar']
    elif sensor == 'sentinel2':
        return config_name_dict['optical']
    else:
        return config_name_dict['fusion']


def subset_activated() -> bool:
    return SETTINGS['SUBSET']['ACTIVATE']


def subset_aois() -> list:
    return SETTINGS['SUBSET']['AOI_IDS']


def input_sensor() -> str:
    return SETTINGS['INPUT']['SENSOR']


def min_timeseries_length() -> bool:
    return SETTINGS['TIMESERIES']['MINIMUM_LENGTH']


def consistent_timeseries_length() -> bool:
    return SETTINGS['TIMESERIES']['CONSISTENT_LENGTH']
