import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from abc import ABC, abstractmethod
from utils import prediction_helpers, dataset_helpers
import scipy


class ChangeDetectionMethod(ABC):

    def __init__(self, name: str):
        self.name = name

    @ abstractmethod
    # returns binary array of changes
    def change_detection(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:
        pass


class ChangeDatingMethod(ChangeDetectionMethod):

    def __init__(self, name: str):
        super().__init__(name)

    @ abstractmethod
    # returns int array where numbers correspond to change date (index in dates list)
    def change_dating(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:
        pass

    @staticmethod
    def _mse(y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.square(y_hat - y), axis=-1) / y.shape[-1]


class StepFunctionModel(ChangeDatingMethod):

    def __init__(self, n_stable: int = 2, max_error: float = 0.5, ts_extension: int = 0):
        super().__init__('stepfunction')
        self.fitted_dataset = None
        self.fitted_aoi = None
        # index when changed occurred in the time series
        # (no change is index 0 and length_ts for non-urban and urban, respectively)
        self.cached_fit = None
        self.length_ts = None
        self.n_stable = n_stable
        self.max_error = max_error
        self.ts_extension = ts_extension

    @staticmethod
    def _nochange_function(x: np.ndarray, return_value: float) -> np.ndarray:
        return np.full(x.shape, fill_value=return_value, dtype=np.float)

    @staticmethod
    def _change_function(x: np.ndarray, x0: float):
        return np.piecewise(x, [x < x0, x >= x0], [0., 1.])


    def _fit(self, dataset: str, aoi_id: str, include_masked_data: bool = False):
        # fit model to aoi id if it's not fit to it
        if dataset == self.fitted_dataset and self.fitted_aoi == aoi_id:
            return

        dates = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)
        self.length_ts = len(dates)

        probs_cube = prediction_helpers.load_prediction_timeseries(dataset, aoi_id, include_masked_data,
                                                                   self.ts_extension)
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
            y_pred_change[:, :, x0 + self.ts_extension:] = 1
            mse_change = self._mse(probs_cube, y_pred_change)
            errors_change.append(mse_change)
        errors_change = np.stack(errors_change)
        min_index_change = np.argmin(errors_change, axis=0)
        min_error_change = np.min(errors_change, axis=0)

        pixels_stable = np.array(min_error_stable < min_error_change)
        self.cached_fit = np.where(pixels_stable, np.zeros(pixels_stable.shape, dtype=np.uint8), min_index_change + 1)

        self.fitted_dataset = dataset
        self.fitted_aoi = aoi_id

    def _predict(self, dataset, aoi_id: str) -> np.ndarray:
        self._fit(dataset, aoi_id)
        y_pred = np.zeros((*self.cached_fit.shape, self.length_ts), dtype=np.uint8)
        # TODO: fix this
        for x0 in range(1, self.length_ts):
            bool_arr = self.cached_fit == x0
            y_pred[bool_arr, x0:] = 1
        return y_pred

    # root mean square error of model
    def model_error(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:
        self._fit(dataset, aoi_id)
        y_pred = self._predict(dataset, aoi_id)
        probs = prediction_helpers.load_prediction_timeseries(dataset, aoi_id, include_masked_data)
        return np.sqrt(self._mse(probs, y_pred))

    def model_confidence(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:
        error = self.model_error(dataset, aoi_id, include_masked_data)
        confidence = np.clip(self.max_error - error, 0, self.max_error) / self.max_error
        return confidence

    def change_detection(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:
        self._fit(dataset, aoi_id, include_masked_data)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = np.logical_and(self.cached_fit != 0, self.cached_fit != self.length_ts)

        return np.array(change).astype(np.uint8)

    def change_dating(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:
        self._fit(dataset, aoi_id)
        change_date = self.cached_fit.copy()
        change_date[change_date == self.length_ts] = 0
        return np.array(change_date).astype(np.uint8)

    # also returns whether a stable pixels is urban or non-urban
    def change_dating_plus(self, aoi_id: str) -> tuple:
        pass


class DeepChangeVectorAnalysis(ChangeDetectionMethod):

    def __init__(self, thresholding_method: str = 'global_otsu', subset_features: bool = False,
                 percentile: int = 90):
        super().__init__('dcva')
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

    def change_detection(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:

        features_start = prediction_helpers.load_features_in_timeseries(dataset, aoi_id, 0, include_masked_data)
        features_end = prediction_helpers.load_features_in_timeseries(dataset, aoi_id, -1, include_masked_data)

        if self.subset_features:
            features_selection = self._get_feature_selection(features_start, features_end)
            features_start = features_start[:, :, features_selection]
            features_end = features_end[:, :, features_selection]

        # compute distance between feature vectors
        magnitude = np.sqrt(np.sum(np.square(features_end - features_start), axis=-1))

        # threshold
        change = self._threshold(magnitude)
        return change


class PostClassificationComparison(ChangeDetectionMethod):

    def __init__(self, threshold: float = 0.5, ignore_negative_changes: bool = False):
        super().__init__('postclassification')
        self.threshold = threshold
        self.ignore_negative_changes = ignore_negative_changes

    def change_detection(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:
        dates = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)
        start_year, start_month, *_ = dates[0]
        end_year, end_month, *_ = dates[-1]
        probs_start = prediction_helpers.load_prediction(dataset, aoi_id, start_year, start_month)
        probs_end = prediction_helpers.load_prediction(dataset, aoi_id, end_year, end_month)
        class_start = probs_start > self.threshold
        class_end = probs_end > self.threshold
        if self.ignore_negative_changes:
            # TODO: test this
            change = np.logical_and(np.logical_not(class_start), class_end)
        else:
            change = class_start != class_end
        return np.array(change).astype(np.uint8)


class Thresholding(ChangeDetectionMethod):

    def __init__(self):
        super().__init__('thresholding')

    def change_detection(self, dataset: str, aoi_id: str) -> np.ndarray:
        dates = dataset_helpers.get_timeseries(dataset, aoi_id)
        start_date = dates[0][:-1]
        end_date = dates[-1][:-1]
        probs_start = prediction_helpers.load_prediction(dataset, aoi_id, *start_date)
        probs_end = prediction_helpers.load_prediction(dataset, aoi_id, *end_date)

        difference = np.abs(probs_end - probs_start)

        thresh = threshold_otsu(difference)
        change = np.array(difference > thresh)

        return np.array(change).astype(np.uint8)


class BreakPointDetection(ChangeDatingMethod):

    def __init__(self, error_multiplier: int = 2, min_prob_diff: float = 0.1, min_segment_length: int = 3,
                 improve_last: bool = False, improve_first: bool = False, noise_reduction: bool = True):
        super().__init__('breakpointdetection')
        self.fitted_dataset = None
        self.fitted_aoi = None
        # index when changed occurred in the time series
        # (no change is index 0 and length_ts for non-urban and urban, respectively)
        self.cached_fit = None
        self.length_ts = None

        self.error_multiplier = error_multiplier
        self.min_prob_diff = min_prob_diff
        self.min_segment_length = min_segment_length
        self.improve_last = improve_last
        self.improve_first = improve_first
        self.noise_reduction = noise_reduction

    def _fit(self, dataset: str, aoi_id: str, include_masked_data: bool):
        if dataset == self.fitted_dataset and self.fitted_aoi == aoi_id:
            return

        timeseries = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)
        self.length_ts = len(timeseries)

        probs_cube = prediction_helpers.load_prediction_timeseries(dataset, aoi_id, include_masked_data)

        errors = []
        mean_diffs = []

        # compute mse for stable fit
        mean_prob = np.mean(probs_cube, axis=-1)
        pred_prob_stable = np.repeat(mean_prob[:, :, np.newaxis], len(timeseries), axis=-1)
        error_stable = self._mse(probs_cube, pred_prob_stable)

        if self.improve_first:
            coefficients = self.exponential_distribution(np.arange(self.length_ts))
            probs_cube = probs_cube.transpose((2, 0, 1))
            probs_cube = coefficients[:, np.newaxis, np.newaxis] * probs_cube
            probs_exp = np.sum(probs_cube, axis=0)
            probs_cube[0, :, :] = probs_exp
            probs_cube = probs_cube.transpose((1, 2, 0))

        if self.improve_last:
            coefficients = self.exponential_distribution(np.arange(self.length_ts))[::-1]
            probs_cube_exp = coefficients[:, np.newaxis, np.newaxis] * probs_cube.transpose((2, 0, 1))
            probs_exp = np.sum(probs_cube_exp, axis=0)
            probs_cube[:, :, -1] = probs_exp


        # break point detection
        for i in range(self.min_segment_length, len(timeseries) - self.min_segment_length):

            # compute predicted
            probs_presegment = probs_cube[:, :, :i]
            mean_prob_presegment = np.mean(probs_presegment, axis=-1)
            pred_probs_presegment = np.repeat(mean_prob_presegment[:, :, np.newaxis], i, axis=-1)

            probs_postsegment = probs_cube[:, :, i:]
            mean_prob_postsegment = np.mean(probs_postsegment, axis=-1)
            pred_probs_postsegment = np.repeat(mean_prob_postsegment[:, :, np.newaxis], len(timeseries) - i, axis=-1)

            # maybe use absolute value here
            mean_diffs.append(mean_prob_postsegment - mean_prob_presegment)

            pred_probs_break = np.concatenate((pred_probs_presegment, pred_probs_postsegment), axis=-1)
            mse_break = self._mse(probs_cube, pred_probs_break)
            errors.append(mse_break)

        errors = np.stack(errors, axis=-1)
        best_fit = np.argmin(errors, axis=-1)

        min_error_break = np.min(errors, axis=-1)
        change_candidate = min_error_break * self.error_multiplier < error_stable

        mean_diffs = np.stack(mean_diffs, axis=-1)
        m, n = mean_diffs.shape[:2]
        mean_diff = mean_diffs[np.arange(m)[:, None], np.arange(n), best_fit]
        change = np.logical_and(change_candidate, mean_diff > self.min_prob_diff)

        if self.noise_reduction:
            kernel = np.ones((3, 3), dtype=np.uint8)
            change_count = scipy.signal.convolve2d(change, kernel, mode='same', boundary='fill', fillvalue=0)
            noise = change_count == 1
            change[noise] = 0

        # self.cached_fit = np.zeros((dataset_helpers.get_yx_size(dataset, aoi_id)), dtype=np.uint8)
        self.cached_fit = np.where(change, best_fit + self.min_segment_length, 0)

        self.fitted_dataset = dataset
        self.fitted_aoi = aoi_id

    def change_detection(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:
        self._fit(dataset, aoi_id, include_masked_data)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = self.cached_fit != 0

        return np.array(change).astype(np.bool)

    def change_dating(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:
        self._fit(dataset, aoi_id, include_masked_data)

        return np.array(self.cached_fit).astype(np.uint8)

    @ staticmethod
    def exponential_distribution(x: np.ndarray, la: float = 0.25) -> np.ndarray:
        return la * np.e ** (-la * x)


class BackwardsBreakPointDetection(ChangeDatingMethod):

    def __init__(self, error_multiplier: int = 2, min_prob_diff: float = 0.1, min_segment_length: int = 3,
                 improved_final_prediction: bool = True):
        super().__init__('backwardsbreakpointdetection')
        self.fitted_dataset = None
        self.fitted_aoi = None
        # index when changed occurred in the time series
        # (no change is index 0 and length_ts for non-urban and urban, respectively)
        self.cached_fit = None
        self.length_ts = None

        self.error_multiplier = error_multiplier
        self.min_prob_diff = min_prob_diff
        self.min_segment_length = min_segment_length
        self.improved_final_prediction = improved_final_prediction

    def _fit(self, dataset: str, aoi_id: str, include_masked_data: bool):
        if dataset == self.fitted_dataset and self.fitted_aoi == aoi_id:
            return

        timeseries = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)
        self.length_ts = len(timeseries)

        probs_cube = prediction_helpers.load_prediction_timeseries(dataset, aoi_id, include_masked_data)

        if self.improved_final_prediction:
            coefficients = self.exponential_distribution(np.arange(self.length_ts))[::-1]
            probs_cube = probs_cube.transpose((2, 0, 1))
            probs_cube = coefficients[:, np.newaxis, np.newaxis] * probs_cube
            probs_exp = np.sum(probs_cube, axis=0)
            probs_cube[-1, :, :] = probs_exp
            probs_cube = probs_cube.transpose((1, 2, 0))

        errors = []
        mean_diffs = []

        # compute mse for stable fit
        mean_prob = np.mean(probs_cube, axis=-1)
        pred_prob_stable = np.repeat(mean_prob[:, :, np.newaxis], len(timeseries), axis=-1)
        error_stable = self._mse(probs_cube, pred_prob_stable)

        pred_last = probs_cube[:, :, -1]

        # break point detection
        for i in range(self.min_segment_length, len(timeseries) - self.min_segment_length):
            # compute predicted
            probs_presegment = probs_cube[:, :, :i]
            mean_prob_presegment = np.mean(probs_presegment, axis=-1)
            pred_probs_presegment = np.repeat(mean_prob_presegment[:, :, np.newaxis], i, axis=-1)

            # maybe use absolute value here
            mean_diffs.append(prob_last - mean_prob_presegment)

            pred_probs_break = np.concatenate((pred_probs_presegment, pred_probs_postsegment), axis=-1)
            mse_break = self._mse(probs_cube, pred_probs_break)
            errors.append(mse_break)

        errors = np.stack(errors, axis=-1)
        best_fit = np.argmin(errors, axis=-1)

        min_error_break = np.min(errors, axis=-1)
        change_candidate = min_error_break * self.error_multiplier < error_stable

        mean_diffs = np.stack(mean_diffs, axis=-1)
        m, n = mean_diffs.shape[:2]
        mean_diff = mean_diffs[np.arange(m)[:, None], np.arange(n), best_fit]
        change = np.logical_and(change_candidate, mean_diff > self.min_prob_diff)

        # self.cached_fit = np.zeros((dataset_helpers.get_yx_size(dataset, aoi_id)), dtype=np.uint8)
        self.cached_fit = np.where(change, best_fit + self.min_segment_length, 0)

        self.fitted_dataset = dataset
        self.fitted_aoi = aoi_id

    def change_detection(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:
        self._fit(dataset, aoi_id, include_masked_data)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = self.cached_fit != 0

        return np.array(change).astype(np.bool)

    def change_dating(self, dataset: str, aoi_id: str, include_masked_data: bool = False) -> np.ndarray:
        self._fit(dataset, aoi_id, include_masked_data)

        return np.array(self.cached_fit).astype(np.uint8)

    @ staticmethod
    def exponential_distribution(x: np.ndarray, la: float = 0.25) -> np.ndarray:
        return la * np.e ** (-la * x)



if __name__ == '__main__':
    ts_length = 5
    x_1d = np.random.rand(ts_length)
    x_3d = np.random.rand(3, 3, ts_length)

    def change_function(x, x0): return np.piecewise(x, [x < x0, x >= x0], [0., 1.])

    y_1d = change_function(x_1d, 2)
    print(y_1d)

    # np.apply_along_axis(my_func, axis=-1, arr=y)

