import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from utils.geofiles import *
import numpy as np
from pathlib import Path
from matplotlib import cm


def plot_optical(ax, file: Path, vis: str = 'true_color', scale_factor: float = 0.4,
                 show_title: bool = False):
    img, _, _ = read_tif(file)
    band_indices = [2, 1, 0] if vis == 'true_color' else [6, 2, 1]
    bands = img[:, :, band_indices] / scale_factor
    bands = bands.clip(0, 1)
    ax.imshow(bands)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title(f'optical ({vis})')


def plot_sar(ax, file: Path, vis: str = 'VV', show_title: bool = False):
    img, _, _ = read_tif(file)
    band_index = 0 if vis == 'VV' else 1
    bands = img[:, :, band_index]
    bands = bands.clip(0, 1)
    ax.imshow(bands, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title(f'sar ({vis})')


def plot_buildings(ax, file: Path, show_title: bool = False):
    img, _, _ = read_tif(file)
    img = img > 0
    img = img if len(img.shape) == 2 else img[:, :, 0]
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('ground truth')


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


def plot_blackwhite(ax, img: np.ndarray, cmap: str = 'gray'):
    ax.imshow(img.clip(0, 1), cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_probability(ax, probability: np.ndarray, title: str = None):
    # ax.imshow(probability, cmap='bwr', vmin=0, vmax=1)
    cmap = colors.ListedColormap(['blue', 'red'])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    # ax.imshow(probability, cmap=cmap, norm=norm)
    # ax.imshow(probability, cmap='Reds', vmin=0, vmax=1.2)
    ax.imshow(probability, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


def plot_prediction(ax, prediction: np.ndarray, show_title: bool = False):
    cmap = colors.ListedColormap(['white', 'red'])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(prediction, cmap=cmap, norm=norm)
    # ax.imshow(prediction, cmap='Reds')
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('prediction')


def plot_probability_histogram(ax, probability: np.ndarray, show_title: bool = False):
    bin_edges = np.linspace(0, 1, 21)
    values = probability.flatten()
    ax.hist(values, bins=bin_edges, range=(0, 1))
    ax.set_xlim((0, 1))
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yscale('log')

    if show_title:
        ax.set_title('probability histogram')


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
