from utils import label_helpers, prediction_helpers, dataset_helpers, visualization, geofiles
import change_detection_models as cd_models
from tqdm import tqdm


def test_stepfunction_model(model: cd_models.ChangeDetectionMethod, aoi_id: str):

    change = model.change_detection(aoi_id)
    change_date = model.change_dating(aoi_id)
    model_error = model.model_error(aoi_id)

    geotransform, crs = dataset_helpers.get_geo(aoi_id)
    save_path = dataset_helpers.root_path() / 'inference' / 'advancedstepfunction'
    save_path.mkdir(exist_ok=True)

    change_file = save_path / f'pred_change_{aoi_id}.tif'
    geofiles.write_tif(change_file, change, geotransform, crs)

    change_date_file = save_path / f'pred_change_date_{aoi_id}.tif'
    geofiles.write_tif(change_date_file, change_date, geotransform, crs)

    model_error_file = save_path / f'model_error_{aoi_id}.tif'
    geofiles.write_tif(model_error_file, model_error, geotransform, crs)


if __name__ == '__main__':

    model = cd_models.ImprovedBasicStepFunctionModel('fusionda_cons05_jaccardmorelikeloss', n_stable=2)

    aoi_ids = dataset_helpers.load_aoi_selection()
    for aoi_id in tqdm(aoi_ids):
        test_stepfunction_model(model, aoi_id)
