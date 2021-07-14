from pathlib import Path
import yaml


def settings() -> dict:
    with open(str(Path.cwd() / 'settings.yaml')) as file:
        s = yaml.load(file, Loader=yaml.FullLoader)
    return s


# dataset names
def spacenet7_dataset_name() -> str:
    s = settings()
    return Path(s['DATASET_NAMES']['SPACENET7'])


def oscd_dataset_name() -> str:
    s = settings()
    return Path(s['DATASET_NAMES']['OSCD'])


# dataset paths
def root_path() -> Path:
    s = settings()
    return Path(s['PATHS']['ROOT_PATH'])


def dataset_path(dataset: str) -> Path:
    return root_path() / dataset_name(dataset)


# path to origin SpaceNet7 dataset
def spacenet7_path() -> Path:
    s = settings()
    return Path(s['PATHS']['SPACENET7_PATH'])


def oscd_path() -> Path:
    s = settings()
    return Path(s['PATHS']['OSCD_PATH'])


def config_name() -> str:
    s = settings()
    sensor = s['INPUT']['SENSOR']
    config_name_dict = s['INPUT']['CONFIG_NAME_DICT']
    if sensor == 'sentinel1':
        return config_name_dict['sar']
    elif sensor == 'sentinel2':
        return config_name_dict['optical']
    else:
        return config_name_dict['fusion']


def include_masked() -> bool:
    s = settings()
    return s['INCLUDE_MASKED_DATA']


def font_size() -> int:
    s = settings()
    return s['PLOTTING']['FONTSIZE']


def plot_size() -> int:
    s = settings()
    return s['PLOTTING']['PLOTSIZE']




