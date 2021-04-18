import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from utils import label, prediction, dataset_helpers, visualization


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
    x = np.linspace(0, n-1, n)
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

    prediction_ts = prediction.generate_timeseries_prediction(config_name, aoi_id)
    y = prediction_ts[i, j, :]
    y_tresh = y > 0.5

    ts = dataset_helpers.get_time_series(aoi_id)
    x = np.linspace(0, len(ts)-1, len(ts))

    print(x, y)
    args = fit_stepfunction(y_tresh)
    print(args)
    fig, ax = plt.subplots()
    y_pred = sigmoid(x, *args)
    visualization.plot_stepfunctionfit(ax, x, y, y_pred, ts, sigmoid, args)


def dummy_data(change_index: int, n: int = 100):
    x = np.arange(n)
    if change_index is None:
        y = np.random.rand(n) + np.random.choice([0, 1])
    else:
        yl = np.random.rand(change_index)
        yr = np.random.rand(n - change_index) + 100
        y = np.concatenate((yl, yr), axis=0) / 100
    return x.astype(np.float), y.astype(np.float)

# https://stackoverflow.com/questions/41147694/how-to-fit-a-step-function-in-python
def plot_heaviside_function():

    x_obs, y_obs = dummy_data(12)
    # yobs = np.random.rand(100) / 100
    plt.plot(x_obs, y_obs, 'ko-')
    plt.ylim((0, 1))
    plt.show()

    def generalized_heaviside_func(x, a, b, c): return a * (np.sign(x - b) + c)
    def change_func(x, a): return 0.5 * (np.sign(x - a) + 1)
    def nochange_func(x, a): return a + x

    popt, pcov = curve_fit(generalized_heaviside_func, x_obs, y_obs)
    # print(popt)

    # popt, pcov = curve_fit(change_func, x_obs, y_obs, bounds=([0], [10]))
    print(popt)


# https://www.semicolonworld.com/question/56035/how-to-apply-piecewise-linear-fit-in-python
def piecewise_function_fit():
    x_obs, y_obs = dummy_data(12)

    def piecewise_linear(x, x0, y0, k1, k2):
        return np.piecewise(x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])

    def change_func(x, x0):
        return np.piecewise(x, [x < x0], [lambda x: 0, lambda x: 1])
    fit_func = change_func
    p, e = curve_fit(fit_func, x_obs, y_obs)
    print(p)
    xd = np.linspace(0, 15, 100)
    plt.plot(x_obs, y_obs, "o")
    plt.plot(xd, fit_func(xd, *p))
    plt.show()


def change_func(x, x0):
    # return np.piecewise(x, [x < a], [lambda x: 0, lambda x: 1])
    return np.piecewise(x, [x < x0, x >= x0], [0., 1.])


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])


def nochange_func(x, a):
    return a


def rmse(y, y_hat):
    return np.sqrt(np.sum(np.square(y_hat - y)) / np.size(y))


def detect_change(x: np.ndarray, y: np.ndarray):
    change_params, _ = curve_fit(change_func, x, y)
    nochange_params, _ = curve_fit(nochange_func, x, y)
    print(change_params, nochange_params)

    y_pred_change = change_func(x, *change_params)
    y_pred_nochange = nochange_func(x, *nochange_params)

    rmse_change = rmse(y, y_pred_change)
    rmse_nochange = rmse(y, y_pred_nochange)


def check_change_detection_algorithm():
    change_index = 50
    x_obs, y_obs = dummy_data(change_index)
    plt.scatter(x_obs, y_obs)
    plt.ylim((0, 1))

    y_pred = change_func(x_obs, change_index)
    plt.plot(x_obs, y_pred, 'ko-')

    plt.show()
    change_params, _ = curve_fit(change_func, x_obs, y_obs)
    print(change_params)
    # detect_change(x_obs, y_obs)


def piecewise_fit2():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=float)
    y = np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])

    def piecewise_linear(x, x0, y0, k1, k2):
        return np.piecewise(x, [x < x0, x >= x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])

    p, e = optimize.curve_fit(piecewise_linear, x, y)
    xd = np.linspace(0, 15, 100)
    plt.plot(x, y, "o")
    plt.plot(xd, piecewise_linear(xd, *p))


if __name__ == '__main__':
    # stepfunction_example()
    # run_stepfunction_on_label('L15-0331E-1257N_1327_3160_13')
    # run_stepfunction_on_prediction('fusionda_cons05_jaccardmorelikeloss', 'L15-0331E-1257N_1327_3160_13')
    # heaviside_stepfunction_example()
    # plot_heaviside_function()
    check_change_detection_algorithm()
    # piecewise_function_fit()