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


def heaviside_stepfunction_example():
    import numpy, scipy, matplotlib
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.optimize import differential_evolution
    import warnings

    # generate data for testing
    x = numpy.linspace(-2, 2, 1000)
    a = 0.5
    yl = numpy.ones_like(x[x < a]) * -0.4 + numpy.random.normal(0, 0.05, x[x < a].shape[0])
    yr = numpy.ones_like(x[x >= a]) * 0.4 + numpy.random.normal(0, 0.05, x[x >= a].shape[0])
    y = numpy.concatenate((yl, yr))

    # alias data to match pervious example
    xData = x
    yData = y

    def func(x, a, b):  # variation of the Heaviside step function
        return 0.5 * b * (numpy.sign(x - a))

    # function for genetic algorithm to minimize (sum of squared error)
    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore")  # do not print warnings by genetic algorithm
        val = func(xData, *parameterTuple)
        return numpy.sum((yData - val) ** 2.0)

    def generate_Initial_Parameters():
        # min and max used for bounds
        maxX = max(xData)
        minX = min(xData)

        parameterBounds = []
        parameterBounds.append([minX, maxX])  # search bounds for a
        parameterBounds.append([minX, maxX])  # search bounds for b

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
        return result.x

    # by default, differential_evolution completes by calling curve_fit() using parameter bounds
    geneticParameters = generate_Initial_Parameters()

    # now call curve_fit without passing bounds from the genetic algorithm,
    # just in case the best fit parameters are aoutside those bounds
    fittedParameters, pcov = curve_fit(func, xData, yData, geneticParameters)
    print('Fitted parameters:', fittedParameters)
    print()

    modelPredictions = func(xData, *fittedParameters)

    absError = modelPredictions - yData

    SE = numpy.square(absError)  # squared errors
    MSE = numpy.mean(SE)  # mean squared errors
    RMSE = numpy.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (numpy.var(absError) / numpy.var(yData))

    print()
    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)

    print()

    ##########################################################
    # graphics output section
    def ModelAndScatterPlot(graphWidth, graphHeight):
        f = plt.figure(figsize=(graphWidth / 100.0, graphHeight / 100.0), dpi=100)
        axes = f.add_subplot(111)

        # first the raw data as a scatter plot
        axes.plot(xData, yData, 'D')

        # create data for the fitted equation plot
        xModel = numpy.linspace(min(xData), max(xData))
        yModel = func(xModel, *fittedParameters)

        # now the model as a line plot
        axes.plot(xModel, yModel)

        axes.set_xlabel('X Data')  # X axis data label
        axes.set_ylabel('Y Data')  # Y axis data label

        plt.show()
        plt.close('all')  # clean up after using pyplot

    graphWidth = 800
    graphHeight = 600
    ModelAndScatterPlot(graphWidth, graphHeight)

def dummy_data(change_index: int, n: int = 100, xlim: tuple = (0, 10)):
    x = np.linspace(*xlim, n)
    if change_index is None:
        y = np.random.rand(n) + np.random.choice([0, 1])
    else:
        yl = np.random.rand(change_index)
        yr = np.random.rand(n - change_index) + 100
        y = np.concatenate((yl, yr), axis=0) / 100
    return x, y

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
    def rmse(y, y_hat): return np.sqrt(np.sum(np.square(y_hat - y)) / np.size(y))
    print(popt)


if __name__ == '__main__':
    # stepfunction_example()
    # run_stepfunction_on_label('L15-0331E-1257N_1327_3160_13')
    # run_stepfunction_on_prediction('fusionda_cons05_jaccardmorelikeloss', 'L15-0331E-1257N_1327_3160_13')
    # heaviside_stepfunction_example()
    plot_heaviside_function()