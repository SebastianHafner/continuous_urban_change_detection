import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from utils import label_helpers, prediction_helpers, dataset_helpers, visualization, geofiles
from change_detection_models import StepFunctionModel
from tqdm import tqdm


def run_stepfunction_on_label(aoi_id: str):

    # find a suitable pixel for time series analysis
    endtoend_label = label_helpers.generate_endtoend_label(aoi_id)
    change_index = 12
    change_on_index = np.argwhere(endtoend_label == change_index)
    pixel_index = 0
    pixel_coords = change_on_index[pixel_index]
    print(pixel_coords)
    i, j, _ = pixel_coords

    assembled_label = label_helpers.generate_timeseries_label(aoi_id)
    # TODO: figure out why list in list
    probs = assembled_label[i, j, :][0]

    dates = dataset_helpers.get_time_series(aoi_id)

    model = StepFunctionModel()
    model.fit(dates, probs)
    pred = model.predict(dates)

    fig, ax = plt.subplots(figsize=(15, 5))
    visualization.plot_fit(ax, dates, probs, pred)
    plt.show()


def run_stepfunction_on_prediction(config_name: str, aoi_id: str):
    # find a suitable pixel for time series analysis
    endtoend_label = label_helpers.generate_endtoend_label(aoi_id)
    change_index = 9
    change_on_index = np.argwhere(endtoend_label == change_index)
    pixel_index = 0
    pixel_coords = change_on_index[pixel_index]
    print(pixel_coords)
    i, j, _ = pixel_coords

    probs_cube = prediction_helpers.generate_timeseries_prediction(config_name, aoi_id)
    probs = np.array([probs_cube[i, j, :]])

    dates = dataset_helpers.get_time_series(aoi_id)

    model = StepFunctionModel()
    model.fit(dates, probs)
    pred = model.predict(dates)

    fig, ax = plt.subplots(figsize=(15, 5))
    visualization.plot_fit(ax, dates, probs, pred, change_index=change_index)
    plt.show()


def run_change_detection(config_name: str, aoi_id: str):

    dates = dataset_helpers.get_time_series(aoi_id)
    probs_cube = prediction_helpers.generate_timeseries_prediction(config_name, aoi_id)
    model = StepFunctionModel()
    change_detection = np.zeros((probs_cube.shape[0], probs_cube.shape[1]), dtype=np.uint8)
    model_error = np.zeros((probs_cube.shape[0], probs_cube.shape[1]), dtype=np.float32)
    n = change_detection.size
    f = 0
    print(change_detection.shape)
    # TODO: maybe get rid of loop
    for index, _ in np.ndenumerate(change_detection):
        i, j = index
        probs = probs_cube[i, j, :]

        model.fit(dates, probs)
        change_detection[index] = model.model + 2
        model_error[index] = model.model_error(dates, probs)
        f += 1
        if f % 10_000 == 0:
            print(f'{f}/{n}')

    geotransform, crs = dataset_helpers.get_geo(aoi_id)
    save_path = dataset_helpers.root_path() / 'inference' / 'stepfunction'
    save_path.mkdir(exist_ok=True)

    change_file = save_path / f'pred_change_{aoi_id}.tif'
    change = change_detection > 1
    geofiles.write_tif(change_file, change.astype(np.uint8), geotransform, crs)

    change_date_file = save_path / f'pred_change_date_{aoi_id}.tif'
    geofiles.write_tif(change_date_file, change_detection, geotransform, crs)

    model_error_file = save_path / f'model_error_{aoi_id}.tif'
    geofiles.write_tif(model_error_file, model_error, geotransform, crs)


if __name__ == '__main__':
    # run_stepfunction_on_label('L15-0331E-1257N_1327_3160_13')
    # run_stepfunction_on_prediction('fusionda_cons05_jaccardmorelikeloss', 'L15-0331E-1257N_1327_3160_13')

    aoi_ids = dataset_helpers.load_aoi_selection()
    for aoi_id in aoi_ids:
        run_change_detection('fusionda_cons05_jaccardmorelikeloss', aoi_id)