from pathlib import Path
from utils import geofiles, visualization, dataset_helpers
import numpy as np


def load_prediction_timeseries(dataset: str, aoi_id: str, ts_extension: int = 0) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(dataset, aoi_id)

    yx_shape = dataset_helpers.get_yx_size(dataset, aoi_id)
    n = len(dates)
    pred_ts = np.zeros((*yx_shape, n + 2 * ts_extension), dtype=np.float32)

    # fill in time series value
    for i, (year, month, *_) in enumerate(dates):
        pred = load_prediction(dataset, aoi_id, year, month)
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


def load_prediction(dataset: str, aoi_id: str, year: int, month: int):
    path = dataset_helpers.dataset_path(dataset) / aoi_id / dataset_helpers.config_name()
    pred_file = path / f'pred_{aoi_id}_{year}_{month:02d}.tif'
    pred, _, _ = geofiles.read_tif(pred_file)
    pred = np.squeeze(pred)
    return pred


def load_prediction_in_timeseries(dataset: str, aoi_id: str, index: int,
                                  ignore_bad_data: bool = True) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(dataset, aoi_id, ignore_bad_data)
    year, month, *_ = dates[index]
    pred = load_prediction(dataset, aoi_id, year, month)
    return pred


def load_features_in_timeseries(dataset: str, aoi_id: str, index: int) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(dataset, aoi_id)
    year, month, _ = dates[index]
    features = load_features(dataset, aoi_id, year, month)
    return features


def load_features(dataset: str, aoi_id: str, year: int, month: int) -> np.ndarray:
    predictions_path = dataset_helpers.dataset_path(dataset) / aoi_id / dataset_helpers.config_name()
    pred_file = predictions_path / f'features_{aoi_id}_{year}_{month:02d}.tif'
    features, _, _ = geofiles.read_tif(pred_file)
    features = np.squeeze(features)
    return features


if __name__ == '__main__':
    predictions = load_prediction_timeseries('spacenet7', 'L15-0331E-1257N_1327_3160_13')
