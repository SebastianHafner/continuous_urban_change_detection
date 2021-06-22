import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from abc import ABC, abstractmethod
from utils import input_helpers, dataset_helpers, geofiles, label_helpers
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm


class ChangeDetectionMethod(ABC):

    def __init__(self, name: str):
        self.name = name

    @ abstractmethod
    # returns binary array of changes
    def change_detection(self, dataset: str, aoi_id: str) -> np.ndarray:
        pass


class ChangeDatingMethod(ChangeDetectionMethod):

    def __init__(self, name: str):
        super().__init__(name)

    @ abstractmethod
    # returns int array where numbers correspond to change date (index in dates list)
    def change_dating(self, dataset: str, aoi_id: str) -> np.ndarray:
        pass

    @staticmethod
    def _mse(y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.square(y_hat - y), axis=-1) / y.shape[-1]


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

    def change_detection(self, dataset: str, aoi_id: str) -> np.ndarray:

        features_start = input_helpers.load_features_in_timeseries(dataset, aoi_id, 0, dataset_helpers.include_masked())
        features_end = input_helpers.load_features_in_timeseries(dataset, aoi_id, -1, dataset_helpers.include_masked())

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

    def change_detection(self, dataset: str, aoi_id: str) -> np.ndarray:
        dates = dataset_helpers.get_timeseries(dataset, aoi_id, dataset_helpers.include_masked())
        start_year, start_month, *_ = dates[0]
        end_year, end_month, *_ = dates[-1]
        probs_start = input_helpers.load_input(dataset, aoi_id, start_year, start_month)
        probs_end = input_helpers.load_input(dataset, aoi_id, end_year, end_month)
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
        dates = dataset_helpers.get_timeseries(dataset, aoi_id, dataset_helpers.include_masked())
        start_date = dates[0][:-1]
        end_date = dates[-1][:-1]
        probs_start = input_helpers.load_input(dataset, aoi_id, *start_date)
        probs_end = input_helpers.load_input(dataset, aoi_id, *end_date)

        difference = np.abs(probs_end - probs_start)

        thresh = threshold_otsu(difference)
        change = np.array(difference > thresh)

        return np.array(change).astype(np.uint8)


class StepFunctionModel(ChangeDatingMethod):

    def __init__(self, error_multiplier: int = 3, min_prob_diff: float = 0.2, min_segment_length: int = 2,
                 improve_last: bool = False, improve_first: bool = False, noise_reduction: bool = True):
        super().__init__('stepfunction')
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

    def _fit(self, dataset: str, aoi_id: str):
        if dataset == self.fitted_dataset and self.fitted_aoi == aoi_id:
            return

        timeseries = dataset_helpers.get_timeseries(dataset, aoi_id, dataset_helpers.include_masked())
        self.length_ts = len(timeseries)

        probs_cube = input_helpers.load_input_timeseries(dataset, aoi_id, dataset_helpers.include_masked())

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

        self.cached_fit = np.where(change, best_fit + self.min_segment_length, 0)

        self.fitted_dataset = dataset
        self.fitted_aoi = aoi_id

    def change_detection(self, dataset: str, aoi_id: str) -> np.ndarray:
        self._fit(dataset, aoi_id)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = self.cached_fit != 0

        return np.array(change).astype(np.bool)

    def change_dating(self, dataset: str, aoi_id: str) -> np.ndarray:
        self._fit(dataset, aoi_id)

        return np.array(self.cached_fit).astype(np.uint8)

    @ staticmethod
    def exponential_distribution(x: np.ndarray, la: float = 0.25) -> np.ndarray:
        return la * np.e ** (-la * x)


class SARStepFunctionModel(StepFunctionModel):

    def __init__(self, config_name: str, error_multiplier: int = 2, min_prob_diff: float = 0.2,
                 min_segment_length: int = 2, improve_last: bool = False, improve_first: bool = False,
                 noise_reduction: bool = True):
        super().__init__(error_multiplier, min_prob_diff=min_prob_diff,
                         min_segment_length=min_segment_length, improve_last=improve_last, improve_first=improve_first,
                         noise_reduction=noise_reduction)
        self.name = 'sarstepfunction'
        self.config_name = config_name

    def change_detection(self, dataset: str, aoi_id: str) -> np.ndarray:
        self._fit(dataset, aoi_id)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = self.cached_fit != 0

        year, month = dataset_helpers.get_date_from_index(-1, dataset, aoi_id, dataset_helpers.include_masked())
        path = dataset_helpers.dataset_path(dataset) / aoi_id / self.config_name
        pred_file = path / f'pred_{aoi_id}_{year}_{month:02d}.tif'
        pred, _, _ = geofiles.read_tif(pred_file)
        is_bua = np.squeeze(pred) > 0.5
        change[~is_bua] = 0

        return np.array(change).astype(np.bool)

    def change_dating(self, dataset: str, aoi_id: str) -> np.ndarray:
        pass


class LogisticFunctionModel(ChangeDatingMethod):

    def __init__(self, min_prob_diff: float = 0.1, noise_reduction: bool = True):
        super().__init__('logisticfunction')
        self.fitted_dataset = None
        self.fitted_aoi = None
        # index when changed occurred in the time series
        # (no change is index 0 and length_ts for non-urban and urban, respectively)
        self.params = None
        self.change_magnitude = None
        self.length_ts = None

        self.min_prob_diff = min_prob_diff
        self.noise_reduction = noise_reduction

    def _fit(self, dataset: str, aoi_id: str):
        if dataset == self.fitted_dataset and self.fitted_aoi == aoi_id:
            return

        timeseries = dataset_helpers.get_timeseries(dataset, aoi_id, dataset_helpers.include_masked())
        self.length_ts = len(timeseries)

        probs_cube = input_helpers.load_input_timeseries(dataset, aoi_id, dataset_helpers.include_masked())
        change_label = label_helpers.generate_change_label(dataset, aoi_id, dataset_helpers.include_masked())

        m, n = dataset_helpers.get_yx_size(dataset, aoi_id)
        params = np.zeros((m, n, 3), dtype=np.float)

        for i in tqdm(range(m)):
            for j in range(n):
                y = probs_cube[i, j, ]
                x = np.arange(1, self.length_ts + 1)
                param_bounds = ([0, 0, 1], [self.length_ts, 1, 4])
                initial_values = [1, 1, 1]
                try:
                    popt, pcov = scipy.optimize.curve_fit(self.logistic_curve, x, y, p0=initial_values,
                                                          bounds=param_bounds)
                    # y_fit = self.logistic_curve(x, *popt)
                    params[i, j, ] = popt
                except RuntimeError:
                    continue

                # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                # ax.plot(x, y, 'o')
                # ax.plot(x, y_fit, '-')
                # ax.set_xlim((0, self.length_ts))
                # ax.set_ylim((0, 1))
                #
                # t0, m, k = popt
                #
                # text_str = f't0={t0:.2f} - m={m:.2f} - k={k:.2f}'
                # ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=14, verticalalignment='top')
                #
                # plt.show()


        self.params = {
            't0': params[:, :, 0],
            'm': params[:, :, 1],
            'k': params[:, :, 2]
        }

        # compute min values
        t = np.full((m, n), fill_value=0., dtype=np.float)
        min_value = self.logistic_curve(t, **self.params)

        t = np.full((m, n), fill_value=self.length_ts, dtype=np.float)
        max_value = self.logistic_curve(t, **self.params)

        self.change_magnitude = max_value - min_value

        # if self.noise_reduction:
        #     kernel = np.ones((3, 3), dtype=np.uint8)
        #     change_count = scipy.signal.convolve2d(change, kernel, mode='same', boundary='fill', fillvalue=0)
        #     noise = change_count == 1
        #     change[noise] = 0

        self.fitted_dataset = dataset
        self.fitted_aoi = aoi_id

    def change_detection(self, dataset: str, aoi_id: str) -> np.ndarray:
        self._fit(dataset, aoi_id)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = self.change_magnitude > self.min_prob_diff

        return np.array(change).astype(np.bool)

    def change_dating(self, dataset: str, aoi_id: str) -> np.ndarray:
        pass

    @staticmethod
    def logistic_curve(t: float, t0: float, m: float, k: float):
        return m / (1 + np.exp(-k * (t - t0)))


if __name__ == '__main__':
    model = LogisticFunctionModel()
    change = model.change_detection('spacenet7', 'L15-0566E-1185N_2265_3451_13')
