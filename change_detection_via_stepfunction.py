import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from utils import label, prediction, dataset_helpers, visualization, dummy_data, geofiles
from change_detection_models import StepFunctionModel
from tqdm import tqdm

def run_stepfunction_on_label(aoi_id: str):

    # find a suitable pixel for time series analysis
    endtoend_label = label.generate_endtoend_label(aoi_id)
    change_index = 12
    change_on_index = np.argwhere(endtoend_label == change_index)
    pixel_index = 0
    pixel_coords = change_on_index[pixel_index]
    print(pixel_coords)
    i, j, _ = pixel_coords

    assembled_label = label.generate_timeseries_label(aoi_id)
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
    endtoend_label = label.generate_endtoend_label(aoi_id)
    change_index = 9
    change_on_index = np.argwhere(endtoend_label == change_index)
    pixel_index = 0
    pixel_coords = change_on_index[pixel_index]
    print(pixel_coords)
    i, j, _ = pixel_coords

    probs_cube = prediction.generate_timeseries_prediction(config_name, aoi_id)
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
    probs_cube = prediction.generate_timeseries_prediction(config_name, aoi_id)
    model = StepFunctionModel()
    change_detection = np.zeros((probs_cube.shape[0], probs_cube.shape[1]), dtype=np.uint8)
    n = change_detection.size
    f = 0
    print(change_detection.shape)
    # TODO: maybe get rid of loop
    for index, _ in np.ndenumerate(change_detection):
        i, j = index
        probs = probs_cube[i, j, :]

        model.fit(dates, probs)

        change_detection[index] = model.model + 2
        f += 1
        if f % 10_000 == 0:
            print(f'{f}/{n}')

    geotransform, crs = dataset_helpers.get_geo(aoi_id)
    cd_file = dataset_helpers.root_path() / 'inference' / 'stepfunction' / f'pred_{aoi_id}.tif'
    cd_file.parent.mkdir(exist_ok=True)
    geofiles.write_tif(cd_file, change_detection, geotransform, crs)


if __name__ == '__main__':
    # run_stepfunction_on_label('L15-0331E-1257N_1327_3160_13')
    # run_stepfunction_on_prediction('fusionda_cons05_jaccardmorelikeloss', 'L15-0331E-1257N_1327_3160_13')
    run_change_detection('fusionda_cons05_jaccardmorelikeloss', 'L15-0331E-1257N_1327_3160_13')