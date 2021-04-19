import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from utils import label, prediction, dataset_helpers, visualization, dummy_data
from change_detection_models import StepFunctionModel


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
    probs = probs_cube[i, j, :]

    dates = dataset_helpers.get_time_series(aoi_id)

    model = StepFunctionModel()
    model.fit(dates, probs)
    pred = model.predict(dates)

    fig, ax = plt.subplots(figsize=(15, 5))
    visualization.plot_fit(ax, dates, probs, pred, change_index=change_index)
    plt.show()


def check_change_detection_algorithm():
    change_index = 10
    x_obs, y_obs = dummy_data(change_index)
    plt.scatter(x_obs, y_obs)
    plt.ylim((0, 1))

    model = StepFunctionModel()
    model.fit(x_obs, y_obs)
    y_pred = model.predict(x_obs)
    plt.plot(x_obs, y_pred, 'k--', label='')
    plt.show()


if __name__ == '__main__':
    # stepfunction_example()
    # run_stepfunction_on_label('L15-0331E-1257N_1327_3160_13')
    # run_stepfunction_on_prediction('fusionda_cons05_jaccardmorelikeloss', 'L15-0331E-1257N_1327_3160_13')
    # heaviside_stepfunction_example()
    # plot_heaviside_function()
    # check_change_detection_algorithm()
    # piecewise_function_fit()
    # piecewise_fit2()
    # run_stepfunction_on_label('L15-0331E-1257N_1327_3160_13')
    run_stepfunction_on_prediction('fusionda_cons05_jaccardmorelikeloss', 'L15-0331E-1257N_1327_3160_13')