from change_detection_models import DeepChangeVectorAnalysis
from utils import dataset_helpers, prediction_helpers


def run_dcva(config_name: str, aoi_id: str):

    model = DeepChangeVectorAnalysis()

    features_first = prediction_helpers.get_features_in_timeseries(config_name, aoi_id, 0)
    features_last = prediction_helpers.get_features_in_timeseries(config_name, aoi_id, -1)
    change_detection = model.detect_changes(features_first, features_last)



if __name__ == '__main__':
    aoi_ids = dataset_helpers.load_aoi_selection()
    config_name = 'fusionda_cons05_jaccardmorelikeloss'
    run_dcva(config_name, aoi_ids[0])
