from pathlib import Path
from utils import geofiles, visualization, dataset_helpers
import numpy as np


def load_prediction_timeseries(config_name: str, dataset: str, aoi_id: str) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(dataset, aoi_id)
    predictions_path = dataset_helpers.root_path() / dataset / aoi_id / config_name
    n = len(dates)
    prediction_ts = None
    for i, (year, month, _) in enumerate(dates):
        pred_file = predictions_path / f'pred_{aoi_id}_{year}_{month:02d}.tif'
        pred, _, _ = geofiles.read_tif(pred_file)
        pred = np.squeeze(pred)
        if prediction_ts is None:
            prediction_ts = np.zeros((*pred.shape, n), dtype=np.float32)
        prediction_ts[:, :, i] = pred

    return prediction_ts


def load_prediction(config_name: str, dataset: str, aoi_id: str, year: int, month: int):
    path = dataset_helpers.root_path() / dataset / aoi_id / config_name
    pred_file = path / f'pred_{aoi_id}_{year}_{month:02d}.tif'
    pred, _, _ = geofiles.read_tif(pred_file)
    return pred


def load_prediction_in_timeseries(config_name: str, dataset: str, aoi_id: str, index: int,
                                  ignore_bad_data: bool = True) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(dataset, aoi_id, ignore_bad_data)
    predictions_path = dataset_helpers.root_path() / dataset / aoi_id / config_name
    year, month, _ = dates[index]
    pred_file = predictions_path / f'pred_{aoi_id}_{year}_{month:02d}.tif'
    pred, _, _ = geofiles.read_tif(pred_file)
    pred = np.squeeze(pred)
    return pred


def load_features_in_timeseries(config_name: str, dataset: str, aoi_id: str, index: int) -> np.ndarray:
    dates = dataset_helpers.get_time_series(dataset, aoi_id)
    predictions_path = dataset_helpers.root_path() / dataset / aoi_id / config_name
    year, month = dates[index]
    pred_file = predictions_path / f'features_{aoi_id}_{year}_{month:02d}.tif'
    features, _, _ = geofiles.read_tif(pred_file)
    features = np.squeeze(features)
    return features


if __name__ == '__main__':
    predictions = load_prediction_timeseries('fusionda_cons05_jaccardmorelikeloss', 'spacenet7_s1s2_dataset',
                                             'L15-0331E-1257N_1327_3160_13')
