import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from utils import geofiles, dataset_helpers
import numpy as np
from pathlib import Path
from matplotlib import cm


class DateColorMap(object):

    def __init__(self, n: int = 25, color_map: str = 'jet'):
        self.n = n
        default_cmap = cm.get_cmap(color_map, n)
        cmap_colors = default_cmap(np.linspace(0, 1, n))
        white = np.array([1, 1, 1, 1])
        black = np.array([0, 0, 0, 1])
        cmap_colors[0, :] = black
        cmap_colors[1, :] = white
        self.cmap = colors.ListedColormap(cmap_colors)

    def get_cmap(self):
        return self.cmap

    def get_vmin(self):
        return 0

    def get_vmax(self):
        return self.n


def plot_optical(ax, aoi_id: str, year: int, month: int, vis: str = 'true_color', scale_factor: float = 0.4):
    file = dataset_helpers.dataset_path() / aoi_id / 'sentinel2' / f'sentinel2_{aoi_id}_{year}_{month:02d}.tif'
    img, _, _ = geofiles.read_tif(file)
    band_indices = [2, 1, 0] if vis == 'true_color' else [6, 2, 1]
    bands = img[:, :, band_indices] / scale_factor
    bands = bands.clip(0, 1)
    ax.imshow(bands)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_sar(ax, aoi_id: str, year: int, month: int, vis: str = 'VV'):
    file = dataset_helpers.dataset_path() / aoi_id / 'sentinel1' / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
    img, _, _ = geofiles.read_tif(file)
    band_index = 0 if vis == 'VV' else 1
    bands = img[:, :, band_index]
    bands = bands.clip(0, 1)
    ax.imshow(bands, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_buildings(ax, aoi_id: str, year: int, month: int):
    file = dataset_helpers.dataset_path() / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
    img, _, _ = geofiles.read_tif(file)
    img = img > 0
    img = img if len(img.shape) == 2 else img[:, :, 0]
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_prediction(ax, config_name: str, aoi_id: str, year: int, month: int):
    file = dataset_helpers.dataset_path() / aoi_id / config_name / f'pred_{aoi_id}_{year}_{month:02d}.tif'
    img, _, _ = geofiles.read_tif(file)
    img = img if len(img.shape) == 2 else img[:, :, 0]
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_endtoend_label(ax, arr: np.ndarray):
    n_colors = 25
    jet = cm.get_cmap('jet', n_colors)
    newcolors = jet(np.linspace(0, 1, n_colors))
    white = np.array([1, 1, 1, 1])
    black = np.array([0, 0, 0, 1])
    newcolors[0, :] = black
    newcolors[1, :] = white
    newcmp = colors.ListedColormap(newcolors)

    ax.imshow(arr, cmap=newcmp, vmin=0, vmax=25)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_change_date(ax, arr: np.ndarray):
    cmap = DateColorMap()
    ax.imshow(arr, cmap=cmap.get_cmap(), vmin=cmap.get_vmin(), vmax=cmap.get_vmax())
    ax.set_xticks([])
    ax.set_yticks([])


def plot_change_data_bar(ax, dates: list):
    cb_ticks = [0.5, 1.5] + list(np.arange(len(dates)) + 2.5)
    cmap = DateColorMap()
    norm = mpl.colors.Normalize(vmin=cmap.get_vmin(), vmax=cmap.get_vmax())
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap.get_cmap(), norm=norm, orientation='horizontal',
                                   ticks=cb_ticks)
    cb.set_label('Change Date (yy-mm)', fontsize=20)
    cb_ticklabels = ['BG', 'BUA'] + [dataset_helpers.date2str(d) for d in dates]
    cb.ax.set_xticklabels(cb_ticklabels)


def plot_blackwhite(ax, img: np.ndarray, cmap: str = 'gray'):
    ax.imshow(img.clip(0, 1), cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_model_error(ax, method: str, aoi_id: str):
    file = dataset_helpers.root_path() / 'inference' / method / f'model_error_{aoi_id}.tif'
    error, _, _ = geofiles.read_tif(file)
    ax.imshow(error, cmap='OrRd', vmin=0, vmax=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_model_error_bar(ax, vmax: float = 0.5):
    cb_ticks = np.linspace(0, vmax, 5)
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cmap = cm.get_cmap('Reds')
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal', ticks=cb_ticks)
    cb.set_label('RMSE', fontsize=20)


def plot_fit(ax, dates: list, probs: np.ndarray, pred: np.ndarray, change_index: int = None):
    x = np.array([year * 12 + month for year, month in dates])
    x_min = np.min(x)
    x = x - x_min
    ax.scatter(x, probs, label='data')
    ax.plot(x, pred, 'k--', label='fit')
    if change_index is not None:
        change_year, change_month = dates[change_index]
        y_change = change_year * 12 + change_month - x_min
        ax.vlines(y_change, ymin=0, ymax=1, colors=['red'], label='change', linestyles='dashed')
    x_labels = [f'{str(year)[-2:]}-{month}' for year, month in dates]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim((-0.1, 1.1))
    ax.legend()


if __name__ == '__main__':
    arr = np.array([[0, 0.01, 0.1, 0.89, 0.9, 1, 1, 1]]).flatten()
    # hist, bin_edges = np.histogram(arr, bins=10, range=(0, 1))
    cmap = mpl.cm.get_cmap('Reds')
    norm = mpl.colors.Normalize(vmin=0, vmax=1.2)

    rgba = cmap(norm(0))
    print(mpl.colors.to_hex(rgba))
    rgba = cmap(norm(1))
    print(mpl.colors.to_hex(rgba))
