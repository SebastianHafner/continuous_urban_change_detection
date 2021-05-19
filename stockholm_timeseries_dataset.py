from utils import dataset_helpers, label_helpers, visualization, metrics
import matplotlib.pyplot as plt
import change_detection_models as cd_models
import numpy as np
from tqdm import tqdm
from pathlib import Path

PATH_DATASET = Path('/storage/shafner/continuous_urban_change_detection/stockholm_timeseries_dataset')


class StockholmTimeseriesInferenceModel(object):

    def __init__(self, config_name: str, n_stable: int = 2, max_error: float = 0.5):

        self.length_ts = None
        self.n_stable = n_stable
        self.max_error = max_error

    @staticmethod
    def _nochange_function(x: np.ndarray, return_value: float) -> np.ndarray:
        return np.full(x.shape, fill_value=return_value, dtype=np.float)

    @staticmethod
    def _change_function(x: np.ndarray, x0: float):
        return np.piecewise(x, [x < x0, x >= x0], [0., 1.])

    @staticmethod
    def _mse(y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.square(y_hat - y), axis=-1) / y.shape[-1]

    def _load_timeseries(self, dates: list, patch_id):

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


def change_detection_inference(model: cd_models.ChangeDetectionMethod, dates: list):

    dates = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)
    start_date = dates[0][:-1]
    end_date = dates[-1][:-1]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # pre image, post image and gt
    visualization.plot_optical(axs[0], dataset, aoi_id, *start_date)
    axs[0].set_title('S2 Start TS')
    visualization.plot_optical(axs[1], dataset, aoi_id, *end_date)
    axs[1].set_title('S2 End TS')

    visualization.plot_change_label(axs[2], dataset, aoi_id, include_masked_data)
    axs[2].set_title('Change GT')

    change = model.change_detection(dataset, aoi_id, include_masked_data)
    visualization.plot_blackwhite(axs[3], change)
    axs[3].set_title('Change Pred')

    if not save_plot:
        plt.show()
    else:
        save_path = dataset_helpers.root_path() / 'plots' / 'testing' / model.name / 'change_detection'
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'change_{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def quantitative_testing(model: cd_models.ChangeDetectionMethod, dataset: str, aoi_id: str,
                         include_masked_data: bool = False):

    # TODO: different quantitative testing for oscd dataset (not penalizing omissions)
    pred = model.change_detection(dataset, aoi_id, include_masked_data)
    gt = label_helpers.generate_change_label(dataset, aoi_id, include_masked_data)

    precision = metrics.compute_precision(pred, gt)
    recall = metrics.compute_recall(pred, gt)
    f1 = metrics.compute_f1_score(pred, gt)

    print(aoi_id)
    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


def quantitative_testing_dataset(model: cd_models.ChangeDetectionMethod, dataset: str,
                                 include_masked_data: bool = False):
    preds, gts = [], []
    for aoi_id in tqdm(dataset_helpers.get_aoi_ids(dataset)):
        pred = model.change_detection(dataset, aoi_id, include_masked_data)
        preds.append(pred.flatten())
        gt = label_helpers.generate_change_label(dataset, aoi_id, include_masked_data)
        gts.append(gt.flatten())
        assert(pred.size == gt.size)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    precision = metrics.compute_precision(preds, gts)
    recall = metrics.compute_recall(preds, gts)
    f1 = metrics.compute_f1_score(preds, gts)

    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


if __name__ == '__main__':
    ds = 'spacenet7'

    dcva = cd_models.DeepChangeVectorAnalysis(subset_features=True)
    pcc = cd_models.PostClassificationComparison()
    thresholding = cd_models.Thresholding()
    stepfunction = cd_models.StepFunctionModel(n_stable=6)

    model = stepfunction
    for aoi_id in dataset_helpers.get_aoi_ids(ds):
        # qualitative_testing(model, ds, aoi_id, save_plot=False)
        # quantitative_testing(model, ds, aoi_id)
        pass
    quantitative_testing_dataset(model, ds, include_masked_data=True)

