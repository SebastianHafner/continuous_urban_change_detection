import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from abc import ABC, abstractmethod
from utils import prediction_helpers, dataset_helpers


class ChangeDetectionMethod(ABC):

    def __init__(self, config_name: str):
        self.config_name = config_name

    @ abstractmethod
    # returns binary array of changes
    def change_detection(self, aoi_id: str) -> np.ndarray:
        pass

    @ abstractmethod
    # returns int array where numbers correspond to change date (index in dates list)
    def change_dating(self, aoi_id: str) -> np.ndarray:
        pass


class BasicStepFunctionModel(ChangeDetectionMethod):

    def __init__(self, config_name: str, n_stable: int = 2):
        super().__init__(config_name)
        self.fitted_aoi = None
        # index when changed occurred in the time series
        # (no change is index 0 and length_ts for non-urban and urban, respectively)
        self.cached_fit = None
        self.length_ts = None
        self.n_stable = n_stable

    @staticmethod
    def _nochange_function(x: np.ndarray, return_value: float) -> np.ndarray:
        return np.full(x.shape, fill_value=return_value, dtype=np.float)

    @staticmethod
    def _change_function(x: np.ndarray, x0: float):
        return np.piecewise(x, [x < x0, x >= x0], [0., 1.])

    @staticmethod
    def _mse(y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.square(y_hat - y), axis=-1) / y.shape[-1]

    def _fit(self, aoi_id: str):
        # fit model to aoi id if it's not fit to it
        if self.fitted_aoi == aoi_id:
            return

        dates = dataset_helpers.get_time_series(aoi_id)
        self.length_ts = len(dates)

        probs_cube = prediction_helpers.generate_timeseries_prediction(self.config_name, aoi_id)
        data_shape = probs_cube.shape

        # fitting stable functions
        stable_values = np.linspace(0, 1, self.n_stable)
        errors_stable = []
        for stable_value in stable_values:
            y_pred_nochange = np.full(data_shape, fill_value=stable_value)
            mse_nochange = self._mse(probs_cube, y_pred_nochange)
            errors_stable.append(mse_nochange)
        errors_stable = np.stack(errors_stable)
        min_error_stable = np.min(errors_stable, axis=0)

        # fitting step functions
        errors_change = []
        for x0 in range(1, self.length_ts):
            y_pred_change = np.zeros(data_shape)
            y_pred_change[:, :, x0:] = 1
            mse_change = self._mse(probs_cube, y_pred_change)
            errors_change.append(mse_change)
        errors_change = np.stack(errors_change)
        min_index_change = np.argmin(errors_change, axis=0)
        min_error_change = np.min(errors_change, axis=0)

        pixels_stable = np.array(min_error_stable < min_error_change)
        self.cached_fit = np.where(pixels_stable, np.zeros(pixels_stable.shape, dtype=np.uint8), min_index_change + 1)
        self.fitted_aoi = aoi_id

    def _predict(self, aoi_id: str) -> np.ndarray:
        self._fit(aoi_id)
        y_pred = np.zeros((*self.cached_fit.shape, self.length_ts), dtype=np.uint8)
        # TODO: this one is probably redundant because 0 initialization
        y_pred[self.cached_fit == 0,] = 0
        y_pred[self.cached_fit == self.length_ts] = 1
        for x0 in range(1, self.length_ts):
            bool_arr = self.cached_fit == x0
            y_pred[bool_arr, x0:] = 1
        return y_pred

    # root mean square error of model
    def model_error(self, aoi_id: str) -> float:
        self._fit(aoi_id)
        y_pred = self._predict(aoi_id)
        probs = prediction_helpers.generate_timeseries_prediction(self.config_name, aoi_id)
        return np.sqrt(self._mse(probs, y_pred))

    def change_detection(self, aoi_id: str) -> np.ndarray:
        self._fit(aoi_id)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = np.logical_and(self.cached_fit != 0, self.cached_fit != self.length_ts)

        return np.array(change).astype(np.uint8)

    def change_dating(self, aoi_id: str) -> np.ndarray:
        self._fit(aoi_id)
        change_date = self.cached_fit.copy()
        change_date[change_date == self.length_ts] = 0
        return np.array(change_date).astype(np.uint8)

    # also returns whether a stable pixels is urban or non-urban
    def change_dating_plus(self, aoi_id: str) -> tuple:
        pass


class AdvancedStepFunctionModel(BasicStepFunctionModel):

    def __init__(self, config_name, max_error: float = 0.25):
        super().__init__(config_name)
        self.max_error = max_error

    def change_detection(self, aoi_id: str) -> np.ndarray:
        self._fit(aoi_id)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = np.logical_and(self.cached_fit != 0, self.cached_fit != self.length_ts)
        model_error = self.model_error(aoi_id)
        change[model_error > self.max_error] = 0

        return np.array(change).astype(np.uint8)

    def change_dating(self, aoi_id: str) -> np.ndarray:
        self._fit(aoi_id)
        change_date = self.cached_fit.copy()
        change_date[change_date == self.length_ts] = 0

        model_error = self.model_error(aoi_id)
        change_date[model_error > self.max_error] = 0

        return np.array(change_date).astype(np.uint8)


class SimplifiedDeepChangeVectorAnalysis(ChangeDetectionMethod):

    def __init__(self, config_name: str, thresholding_method: str = 'global_otsu', subset_features: bool = False,
                 percentile: int = 90):
        super().__init__(config_name)
        self.thresholding_method = thresholding_method
        self.subset_features = subset_features
        self.percentile = percentile

    def _get_feature_selection(self, features_t1: np.ndarray, features_t2: np.ndarray) -> list:
        diff = features_t2 - features_t1
        var = np.var(diff, axis=(0, 1))
        indices_sorted = list(np.argsort(var))[::-1]

        # percentile
        n_features = diff.shape[-1]
        n_selection = n_features // 100 * (100 - self.percentile)
        return indices_sorted[:n_selection]

    def _threshold(self, arr: np.ndarray) -> np.ndarray:
        if self.thresholding_method == 'local_adaptive':
            block_size = 35
            binary = threshold_local(arr, block_size, offset=10)
        elif self.thresholding_method == 'global_otsu':
            thresh = threshold_otsu(arr)
            binary = arr > thresh
        else:
            raise Exception('Unknown thresholding method')
        return binary.astype(np.uint8)

    def change_detection(self, aoi_id: str) -> np.ndarray:

        features_start = prediction_helpers.get_features_in_timeseries(self.config_name, aoi_id, 0)
        features_end = prediction_helpers.get_features_in_timeseries(self.config_name, aoi_id, -1)

        if self.subset_features:
            features_selection = self._get_feature_selection(features_start, features_end)
            features_start = features_start[:, :, features_selection]
            features_end = features_end[:, :, features_selection]

        # compute distance between feature vectors
        magnitude = np.sqrt(np.sum(np.square(features_end - features_start), axis=-1))

        # threshold
        change = self._threshold(magnitude)
        return change

    def change_dating(self, aoi_id: str) -> np.ndarray:
        pass


class PostClassificationComparison(ChangeDetectionMethod):

    def __init__(self, config_name: str, threshold: float = 0.5, ignore_negative_changes: bool = False):
        super().__init__(config_name)
        self.threshold = threshold
        self.ignore_negative_changes = ignore_negative_changes

    def change_detection(self, aoi_id: str) -> np.ndarray:
        probs_start = prediction_helpers.get_prediction_in_timeseries(self.config_name, aoi_id, 0)
        probs_end = prediction_helpers.get_prediction_in_timeseries(self.config_name, aoi_id, -1)
        class_start = probs_start > self.threshold
        class_end = probs_end > self.threshold
        if self.ignore_negative_changes:
            # TODO: test this
            change = np.logical_and(np.logical_not(class_start), class_end)
        else:
            change = class_start != class_end
        return np.array(change).astype(np.uint8)

    def change_dating(self, aoi_id: str) -> np.ndarray:
        pass

if __name__ == '__main__':
    ts_length = 5
    x_1d = np.random.rand(ts_length)
    x_3d = np.random.rand(3, 3, ts_length)

    def change_function(x, x0): return np.piecewise(x, [x < x0, x >= x0], [0., 1.])

    y_1d = change_function(x_1d, 2)
    print(y_1d)

    # np.apply_along_axis(my_func, axis=-1, arr=y)

