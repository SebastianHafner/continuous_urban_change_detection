from change_detection_models import DeepChangeVectorAnalysis
from utils import dataset_helpers, prediction_helpers, geofiles


def run_dcva(config_name: str, aoi_id: str):

    model = DeepChangeVectorAnalysis(subset_features=True)

    features_first = prediction_helpers.get_features_in_timeseries(config_name, aoi_id, 0)
    features_last = prediction_helpers.get_features_in_timeseries(config_name, aoi_id, -1)
    change = model.detect_changes(features_first, features_last)
    geotransform, crs = dataset_helpers.get_geo(aoi_id)
    cd_file = dataset_helpers.root_path() / 'inference' / 'dcva' / f'pred_{aoi_id}.tif'
    cd_file.parent.mkdir(exist_ok=True)
    geofiles.write_tif(cd_file, change, geotransform, crs)


if __name__ == '__main__':
    config_name = 'fusionda_cons05_jaccardmorelikeloss'
    for aoi_id in dataset_helpers.load_aoi_selection():
        print(aoi_id)
        run_dcva(config_name, aoi_id)
