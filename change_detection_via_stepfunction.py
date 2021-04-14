import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from utils import label, prediction, dataset_helpers


def sigmoid(x, x0, b):
    return scipy.special.expit((x - x0) * b)


def stepfunction_example():
    x = np.linspace(0,10,101)
    y = np.heaviside((x-5), 0.)

    args, cov = curve_fit(sigmoid, x, y)
    print(args)
    plt.scatter(x,y)
    plt.plot(x, sigmoid(x, *args))
    plt.show()


def fit_stepfunction(probs: list):
    n = len(probs)
    x = np.linspace(0, n, n)
    args, cov = curve_fit(sigmoid, x, probs)
    return args


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
    y = assembled_label[i, j, :][0]

    ts = dataset_helpers.get_time_series(aoi_id)
    x = [year * 12 + month for year, month in ts]
    x = np.linspace(0, len(x), len(x))

    print(x, y)
    args = fit_stepfunction(y)
    print(args)
    plt.scatter(x,y)
    plt.plot(x, sigmoid(x, *args))
    plt.show()
    pass


def run_stepfunction_on_prediction(config_name: str, aoi_id: str):
    # find a suitable pixel for time series analysis
    endtoend_label = label.generate_endtoend_label(aoi_id)
    change_index = 10
    change_on_index = np.argwhere(endtoend_label == change_index)
    pixel_index = 0
    pixel_coords = change_on_index[pixel_index]
    print(pixel_coords)
    i, j, _ = pixel_coords

    prediction_ts = prediction.generate_timeseries_prediction('fusionda_cons05_jaccardmorelikeloss', aoi_id)
    y = prediction_ts[i, j, :]

    ts = dataset_helpers.get_time_series(aoi_id)
    x = [year * 12 + month for year, month in ts]
    x = np.linspace(0, len(x), len(x))

    print(x, y)
    args = fit_stepfunction(y)
    print(args)
    plt.scatter(x, y)
    plt.plot(x, sigmoid(x, *args))
    plt.show()
    pass


if __name__ == '__main__':
    # stepfunction_example()
    # run_stepfunction_on_label('L15-0331E-1257N_1327_3160_13')
    run_stepfunction_on_prediction('fusionda_cons05_jaccardmorelikeloss', 'L15-0331E-1257N_1327_3160_13')
