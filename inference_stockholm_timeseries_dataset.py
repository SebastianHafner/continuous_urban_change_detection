from utils import geofiles
import numpy as np
from pathlib import Path
from tqdm import tqdm


class StockholmTimeseriesModel(object):

    def __init__(self, dataset_path: Path, config_name: str, n_stable: int = 6, patch_size: int = 256):
        self.dataset_path = dataset_path
        self.config_name = config_name
        self.cached_fit = None
        self.length_ts = None
        self.fitted_patch_id = None
        self.n_stable = n_stable
        self.patch_size = patch_size

    @staticmethod
    def _nochange_function(x: np.ndarray, return_value: float) -> np.ndarray:
        return np.full(x.shape, fill_value=return_value, dtype=np.float)

    @staticmethod
    def _change_function(x: np.ndarray, x0: float):
        return np.piecewise(x, [x < x0, x >= x0], [0., 1.])

    @staticmethod
    def _mse(y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.square(y_hat - y), axis=-1) / y.shape[-1]

    def _load_timeseries(self, dates: list, patch_id: str):
        prob_cube = np.zeros((self.patch_size, self.patch_size, len(dates)), dtype=np.float32)
        for i, date in enumerate(dates):
            pred_path = self.dataset_path / date / self.config_name
            file = pred_path / f'prob_stockholm_{date}_{patch_id}.tif'
            prob, _, _ = geofiles.read_tif(file, first_band_only=True)
            prob_cube[:, :, i] = prob
        return prob_cube

    def _fit(self, dates: list, patch_id: str):
        if self.fitted_patch_id == patch_id:
            return
        else:
            self.fitted_patch_id = patch_id

        self.length_ts = len(dates)

        probs_cube = self._load_timeseries(dates, patch_id)
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

    def change_detection(self, dates: list, patch_id: str) -> np.ndarray:
        self._fit(dates, patch_id)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = np.logical_and(self.cached_fit != 0, self.cached_fit != self.length_ts)

        return np.array(change).astype(np.uint8)

    def change_dating(self, dates: list, patch_id: str) -> np.ndarray:
        self._fit(dates, patch_id)
        change_date = self.cached_fit.copy()
        change_date[change_date == self.length_ts] = 0
        return np.array(change_date).astype(np.uint8)

    # also returns whether a stable pixels is urban or non-urban
    def change_dating_plus(self, dates: list, patch_id: str) -> tuple:
        pass


def get_timeseries() -> list:
    dates = ['2015-08', '2015-11', '2016-04', '2016-07', '2016-10', '2017-04', '2017-07', '2017-10', '2018-04',
             '2018-06', '2018-10', '2019-04', '2019-06', '2019-09', '2020-03', '2020-06', '2020-09', '2021-04']
    return sorted(dates)


def get_patch_ids(dataset_file: Path, date: str = '2015-08'):
    samples_file = dataset_file / f'samples_{date}.json'
    samples = geofiles.load_json(samples_file)
    patch_ids = [s['patch_id'] for s in samples['samples']]
    return sorted(patch_ids)


def get_geo(dataset_file: Path, date: str = '2015-08') -> tuple:
    tl_file = dataset_file / date / 'sentinel1' / f'sentinel1_stockholm_{date}_{0:010d}-{0:010d}.tif'
    _, transform, crs = geofiles.read_tif(tl_file)
    return transform, crs


def get_yx_size(patch_ids: list, patch_size: int = 256):
    bl_patch = max(patch_ids)
    y, x = bl_patch.split('-')
    y_max, x_max = int(y) + patch_size, int(x) + patch_size
    return y_max, x_max


def get_index(patch_id: str):
    y, x = patch_id.split('-')
    y_index, x_index = int(y),  int(x)
    return y_index, x_index


def run_inference():
    path = Path('/storage/shafner/continuous_urban_change_detection/stockholm_timeseries_dataset')
    config_name = 'fusionda_cons05_jaccardmorelikeloss'

    model = StockholmTimeseriesModel(path, config_name)

    dates = get_timeseries()
    patch_ids = get_patch_ids(path)

    y_max, x_max = get_yx_size(patch_ids)
    transform, crs = get_geo(path)

    change = np.zeros((y_max, x_max), dtype=np.uint8)
    change_date = np.zeros((y_max, x_max), dtype=np.uint8)

    for patch_id in tqdm(patch_ids):
        change_patch = model.change_detection(dates, patch_id)
        change_date_patch = model.change_detection(dates, patch_id)
        y_index, x_index = get_index(patch_id)
        change[y_index:y_index+256, x_index:x_index+256] = change_patch
        change_date[y_index:y_index + 256, x_index:x_index + 256] = change_date_patch

    save_path = path / 'change'
    save_path.mkdir(exist_ok=True)
    change_file = save_path / 'change_stockholm.tif'
    geofiles.write_tif(change_file, change, transform, crs)

    change_date_file = save_path / 'change_date_stockholm.tif'
    geofiles.write_tif(change_date_file, change_date, transform, crs)

if __name__ == '__main__':
    run_inference()
