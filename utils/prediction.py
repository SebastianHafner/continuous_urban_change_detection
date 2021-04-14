from pathlib import Path
from utils import geofiles, visualization, dataset_helpers
import numpy as np


def generate_timeseries_prediction(config_name: str, aoi_id: str) -> np.ndarray:
    dates = dataset_helpers.get_time_series(aoi_id)
    predictions_path = dataset_helpers.dataset_path() / aoi_id / config_name
    n = len(dates)
    prediction_ts = None
    for i, (year, month) in enumerate(dates):
        pred_file = predictions_path / f'pred_{aoi_id}_{year}_{month:02d}.tif'
        pred, _, _ = geofiles.read_tif(pred_file)
        pred = np.squeeze(pred)
        if prediction_ts is None:
            prediction_ts = np.zeros((*pred.shape, n), dtype=np.float32)
        prediction_ts[:, :, i] = pred

    return prediction_ts


if __name__ == '__main__':
    predictions = generate_timeseries_prediction('fusionda_cons05_jaccardmorelikeloss', 'L15-0331E-1257N_1327_3160_13')
