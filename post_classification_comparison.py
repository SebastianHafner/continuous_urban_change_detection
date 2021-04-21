from utils import geofiles, dataset_helpers, prediction_helpers
from change_detection_models import PostClassificationComparison
import numpy as np


def run_change_detection(config_name: str, aoi_id: str):
    probs_start = prediction_helpers.get_prediction_in_timeseries(config_name, aoi_id, 0)
    probs_end = prediction_helpers.get_prediction_in_timeseries(config_name, aoi_id, -1)

    model = PostClassificationComparison()
    change = model.detect_changes(probs_start, probs_end)
    geotransform, crs = dataset_helpers.get_geo(aoi_id)
    cd_file = dataset_helpers.root_path() / 'inference' / 'postcomparison' / f'pred_{aoi_id}.tif'
    cd_file.parent.mkdir(exist_ok=True)
    geofiles.write_tif(cd_file, change, geotransform, crs)
