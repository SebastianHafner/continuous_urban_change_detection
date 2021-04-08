import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def stepfunction_example():
    x = np.linspace(0,10,101)
    y = np.heaviside((x-5), 0.)

    def sigmoid(x, x0,b):
        return scipy.special.expit((x-x0)*b)

    args, cov = curve_fit(sigmoid, x, y)
    plt.scatter(x,y)
    plt.plot(x, sigmoid(x, *args))
    plt.show()
    print(args)


if __name__ == '__main__':
    stepfunction_example()
