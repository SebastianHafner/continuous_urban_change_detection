from pathlib import Path
from utils import geofiles, dataset_helpers, config
import numpy as np


def input_name() -> str:
    return f'{config.input_type()}_{config.input_sensor()}'


def load_input(aoi_id: str, year: int, month: int) -> np.ndarray:
    return load_prediction(aoi_id, year, month)


def load_sentinel1(dataset: str, aoi_id: str, year: int, month: int, band: str):
    file = dataset_helpers.dataset_path(dataset) / aoi_id / 'sentinel1' / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
    img, _, _ = geofiles.read_tif(file)
    bands = ['VV', 'VH']
    band_index = bands.index(band)
    img = img[:, :, band_index]
    return img


def load_sentinel2(dataset: str, aoi_id: str, year: int, month: int, band: str):
    file = dataset_helpers.dataset_path(dataset) / aoi_id / 'sentinel2' / f'sentinel2_{aoi_id}_{year}_{month:02d}.tif'
    img, _, _ = geofiles.read_tif(file)
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    band_index = bands.index(band)
    img = img[:, :, band_index]
    return img


def load_prediction_raw(aoi_id: str, year: int, month: int, config_name: str) -> np.ndarray:
    path = dataset_helpers.dataset_path() / aoi_id / config_name
    pred_file = path / f'pred_{aoi_id}_{year}_{month:02d}.tif'
    pred, _, _ = geofiles.read_tif(pred_file)
    pred = np.squeeze(pred)
    return pred


def load_prediction(aoi_id: str, year: int, month: int) -> np.ndarray:
    pred = load_prediction_raw(aoi_id, year, month, config.config_name())
    return pred


def load_prediction_in_timeseries(aoi_id: str, index: int) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(dataset, aoi_id)
    year, month, *_ = dates[index]
    pred = load_prediction(aoi_id, year, month)
    return pred


def prediction_is_available(aoi_id: str, year: int, month: int) -> bool:
    path = dataset_helpers.dataset_path() / aoi_id / config.config_name()
    pred_file = path / f'pred_{aoi_id}_{year}_{month:02d}.tif'
    return pred_file.exists()


def load_input_timeseries(aoi_id: str, ts_extension: int = 0) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(aoi_id)

    yx_shape = dataset_helpers.get_yx_size(aoi_id)
    n = len(dates)
    pred_ts = np.zeros((*yx_shape, n + 2 * ts_extension), dtype=np.float32)

    # fill in time series value
    for i, (year, month, *_) in enumerate(dates):
        pred = load_input(aoi_id, year, month)
        pred_ts[:, :, i + ts_extension] = pred

    # padd start and end
    if ts_extension != 0:
        start_pred = pred_ts[:, :, ts_extension]
        start_extension = np.repeat(start_pred[:, :, np.newaxis], ts_extension, axis=2)
        pred_ts[:, :, :ts_extension] = start_extension

        end_index = ts_extension + n - 1
        end_pred = pred_ts[:, :, end_index]
        end_extension = np.repeat(end_pred[:, :, np.newaxis], ts_extension, axis=2)
        pred_ts[:, :, end_index + 1:] = end_extension

    return pred_ts


def load_input_in_timeseries(aoi_id: str, index: int) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(aoi_id)
    year, month, *_ = dates[index]
    pred = load_prediction(aoi_id, year, month)
    return pred


if __name__ == '__main__':
    # predictions = load_prediction_timeseries('spacenet7', 'L15-0331E-1257N_1327_3160_13')
    a = np.array([False, np.NaN])
    a = np.ma.array(a, mask=np.isnan(a))
    print(np.any(a))
