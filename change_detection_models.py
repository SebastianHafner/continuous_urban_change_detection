import numpy as np


class StepFunctionModel(object):

    def __init__(self):
        self.model = None

    @staticmethod
    def nochange_function(x: np.ndarray, return_value: float) -> np.ndarray:
        return np.full(x.shape, fill_value=return_value, dtype=np.float)

    @staticmethod
    def change_function(x, x0):
        return np.piecewise(x, [x < x0, x >= x0], [0., 1.])

    @staticmethod
    def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.square(y_hat - y)) / np.size(y)

    # root mean square error of model
    def model_error(self, x_obs, y_obs) -> float:
        y_pred = self.predict(x_obs)
        return np.sqrt(self.mse(y_obs, y_pred))

    def fit(self, dates: list, probs: np.ndarray):
        x = np.arange(len(dates))
        y_pred_nochange_nourban = self.nochange_function(x, 0)
        mse_nochange_nourban = self.mse(probs, y_pred_nochange_nourban)
        y_pred_nochange_urban = self.nochange_function(x, 1)
        mse_nochange_urban = self.mse(probs, y_pred_nochange_urban)

        errors = [mse_nochange_nourban, mse_nochange_urban]
        for x0 in x:
            y_pred_change = self.change_function(x, x0)
            mse_change = self.mse(probs, y_pred_change)
            errors.append(mse_change)

        index = np.argmin(errors)
        self.model = index - 2

    def predict(self, dates: list) -> np.ndarray:
        x = np.arange(len(dates))
        if self.model == -2:
            return self.nochange_function(x, 0.)
        elif self.model == -1:
            return self.nochange_function(x, 1.)
        else:
            return self.change_function(x, self.model)

    def is_fitted(self) -> bool:
        return False if self.model is None else True


class DeepChangeVectorAnalysis(object):

    def __init__(self):
        pass
