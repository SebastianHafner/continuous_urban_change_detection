from utils import geofiles, dataset_helpers, prediction
import numpy as np


def run_change_detection(config_name: str, aoi_id: str):
    probs_start = prediction.get_prediction_in_timeseries(config_name, aoi_id, 0)
    probs_end = prediction.get_prediction_in_timeseries(config_name, aoi_id, -1)

    pred_start = probs_start > 0.5
    pred_end = probs_end > 0.5

    change_detection = pred_start != pred_end
    geotransform, crs = dataset_helpers.get_geo(aoi_id)
    cd_file = dataset_helpers.root_path() / 'inference' / 'postcomparison' / f'pred_{aoi_id}.tif'
    cd_file.parent.mkdir(exist_ok=True)
    geofiles.write_tif(cd_file, change_detection, geotransform, crs)
