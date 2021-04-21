import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from abc import ABC, abstractmethod


class ChangeDetectionMethod(ABC):

    def __init__(self):
        pass

    @ abstractmethod
    def detect_changes(self, input_t1: np.ndarray, input_t2: np.ndarray) -> np.ndarray:
        pass


class StepFunctionModel(ChangeDetectionMethod):

    def __init__(self):
        super().__init__()
        self.model = None

    @staticmethod
    def nochange_function(x: np.ndarray, return_value: float) -> np.ndarray:
        return np.full(x.shape, fill_value=return_value, dtype=np.float)

    @staticmethod
    def change_function(x: np.ndarray, x0: float):
        return np.piecewise(x, [x < x0, x >= x0], [0., 1.])

    @staticmethod
    def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.square(y_hat - y)) / np.size(y)

    # root mean square error of model
    def model_error(self, dates, probs) -> float:
        assert(self.is_fitted())
        y_pred = self.predict(dates)
        return np.sqrt(self.mse(probs, y_pred))

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

    def detect_changes(self, input_t1: np.ndarray, input_t2: np.ndarray) -> np.ndarray:
        pass


class DeepChangeVectorAnalysis(ChangeDetectionMethod):

    def __init__(self, thresholding_method: str = 'global_otsu', subset_features: bool = False,
                 percentile: int = 90):
        super().__init__()
        self.thresholding_method = thresholding_method
        self.subset_features = subset_features
        self.percentile = percentile

    def detect_changes(self, features_t1: np.ndarray, features_t2: np.ndarray) -> np.ndarray:

        if self.subset_features:
            features_selection = self.get_feature_selection(features_t1, features_t2)
            features_t1 = features_t1[:, :, features_selection]
            features_t2 = features_t2[:, :, features_selection]

        # compute distance between feature vectors
        magnitude = np.sqrt(np.sum(np.square(features_t2 - features_t1), axis=-1))

        # threshold
        change = self.threshold(magnitude)
        return change

    def get_feature_selection(self, features_t1: np.ndarray, features_t2: np.ndarray) -> list:
        diff = features_t2 - features_t1
        var = np.var(diff, axis=(0, 1))
        indices_sorted = list(np.argsort(var))[::-1]

        # percentile
        n_features = diff.shape[-1]
        n_selection = n_features // 100 * (100 - self.percentile)
        return indices_sorted[:n_selection]

    def threshold(self, arr: np.ndarray) -> np.ndarray:
        if self.thresholding_method == 'local_adaptive':
            block_size = 35
            binary = threshold_local(arr, block_size, offset=10)
        elif self.thresholding_method == 'global_otsu':
            thresh = threshold_otsu(arr)
            binary = arr > thresh
        else:
            raise Exception('Unknown thresholding method')
        return binary.astype(np.uint8)


class PostClassificationComparison(ChangeDetectionMethod):

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def detect_changes(self, probs_t1: np.ndarray, probs_t2: np.ndarray) -> np.ndarray:
        class_t1 = probs_t1 > self.threshold
        class_t2 = probs_t2 > self.threshold
        change = class_t1 != class_t2
        return np.array(change).astype(np.uint8)
