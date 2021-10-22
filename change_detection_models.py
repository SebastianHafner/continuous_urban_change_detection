import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from abc import ABC, abstractmethod
from utils import input_helpers, dataset_helpers, geofiles, label_helpers, config
import scipy
from tqdm import tqdm


class ChangeDetectionMethod(ABC):

    def __init__(self, name: str):
        self.name = name

    @ abstractmethod
    # returns binary array of changes
    def change_detection(self, aoi_id: str) -> np.ndarray:
        pass


class ChangeDatingMethod(ChangeDetectionMethod):

    def __init__(self, name: str):
        super().__init__(name)

    @ abstractmethod
    # returns int array where numbers correspond to change date (index in dates list)
    def change_dating(self, aoi_id: str) -> np.ndarray:
        pass

    @staticmethod
    def _mse(y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.square(y_hat - y), axis=-1) / y.shape[-1]


class PostClassificationComparison(ChangeDetectionMethod):

    def __init__(self, threshold: float = 0.5, ignore_negative_changes: bool = False):
        super().__init__('postclassification')
        self.threshold = threshold
        self.ignore_negative_changes = ignore_negative_changes

    def change_detection(self, aoi_id: str) -> np.ndarray:
        dates = dataset_helpers.get_timeseries(aoi_id)
        start_year, start_month, *_ = dates[0]
        end_year, end_month, *_ = dates[-1]
        probs_start = input_helpers.load_input(aoi_id, start_year, start_month)
        probs_end = input_helpers.load_input(aoi_id, end_year, end_month)
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

    def change_detection(self, aoi_id: str) -> np.ndarray:
        dates = dataset_helpers.get_timeseries(aoi_id)
        start_date = dates[0][:-1]
        end_date = dates[-1][:-1]
        probs_start = input_helpers.load_input(aoi_id, *start_date)
        probs_end = input_helpers.load_input(aoi_id, *end_date)

        difference = np.abs(probs_end - probs_start)

        thresh = threshold_otsu(difference)
        change = np.array(difference > thresh)

        return np.array(change).astype(np.uint8)


class StepFunctionModel(ChangeDatingMethod):

    def __init__(self, error_multiplier: int = 2, min_prob_diff: float = 0.35, min_segment_length: int = 2):
        super().__init__('stepfunction')
        self.fitted_aoi = None
        # index when changed occurred in the time series
        # (no change is index 0 and length_ts for non-urban and urban, respectively)
        self.cached_fit = None
        self.length_ts = None

        self.error_multiplier = error_multiplier
        self.min_prob_diff = min_prob_diff
        self.min_segment_length = min_segment_length

    def _fit(self, aoi_id: str):
        if self.fitted_aoi == aoi_id:
            return

        timeseries = dataset_helpers.get_timeseries(aoi_id)
        self.length_ts = len(timeseries)

        probs_cube = input_helpers.load_input_timeseries(aoi_id)

        errors = []
        mean_diffs = []

        # compute mse for stable fit
        mean_prob = np.mean(probs_cube, axis=-1)
        pred_prob_stable = np.repeat(mean_prob[:, :, np.newaxis], len(timeseries), axis=-1)
        error_stable = self._mse(probs_cube, pred_prob_stable)

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

        self.cached_fit = np.where(change, best_fit + self.min_segment_length, 0)

        self.fitted_aoi = aoi_id

    def change_detection(self, aoi_id: str, noise_reduction: bool = False) -> np.ndarray:
        self._fit(aoi_id)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = self.cached_fit != 0
        # kernel = np.ones((5, 5), dtype=np.uint8)
        # change_count = scipy.signal.convolve2d(change, kernel, mode='same', boundary='fill', fillvalue=0)
        # noise = change_count == 1
        # change[noise] = 0

        return np.array(change).astype(np.bool)

    def change_dating(self, aoi_id: str, config_name: str = None) -> np.ndarray:
        self._fit(aoi_id)

        return np.array(self.cached_fit).astype(np.uint8)


class KernelStepFunctionModel(StepFunctionModel):

    def __init__(self, kernel_size: int = 3, error_multiplier: int = 2, min_prob_diff: float = 0.35,
                 min_segment_length: int = 2):
        super().__init__(error_multiplier, min_prob_diff, min_segment_length)

        self.kernel_size = kernel_size

    def _fit(self, aoi_id: str):
        if self.fitted_aoi == aoi_id:
            return

        timeseries = dataset_helpers.get_timeseries(aoi_id)
        self.length_ts = len(timeseries)

        probs_cube_raw = input_helpers.load_input_timeseries(aoi_id)
        kernel = np.full((self.kernel_size, self.kernel_size), fill_value=1/self.kernel_size**2)
        probs_cube = np.empty(probs_cube_raw.shape, dtype=np.single)
        for i in range(self.length_ts):
            probs_cube[:, :, i] = scipy.signal.convolve2d(probs_cube_raw[:, :, i], kernel, mode='same', boundary='symm')

        errors = []
        mean_diffs = []

        # compute mse for stable fit
        mean_prob = np.mean(probs_cube, axis=-1)
        pred_prob_stable = np.repeat(mean_prob[:, :, np.newaxis], len(timeseries), axis=-1)
        error_stable = self._mse(probs_cube, pred_prob_stable)

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

        self.cached_fit = np.where(change, best_fit + self.min_segment_length, 0)

        self.fitted_aoi = aoi_id



if __name__ == '__main__':
    pass
