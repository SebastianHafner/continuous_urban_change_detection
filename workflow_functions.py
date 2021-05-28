from pathlib import Path
from utils import geofiles, visualization, dataset_helpers, prediction_helpers, label_helpers, metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import change_detection_models as cd_models


def produce_workflow_data(dataset: str, aoi_id: str):

    md = dataset_helpers.aoi_metadata(dataset, aoi_id)

    for i, (year, month, mask, s1, s2) in enumerate(tqdm(md)):

        if s1 and s2:

            # make sentinel 1 figure
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            visualization.plot_sar(ax, dataset, aoi_id, year, month)
            output_path = dataset_helpers.root_path() / 'plots' / 'workflow' / 'sentinel1'
            output_file = output_path / f'sentinel1_{aoi_id}_{year}_{month:02d}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

            # make sentinel 2 figure
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            visualization.plot_optical(ax, dataset, aoi_id, year, month)
            output_path = dataset_helpers.root_path() / 'plots' / 'workflow' / 'sentinel2'
            output_file = output_path / f'sentinel2_{aoi_id}_{year}_{month:02d}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True,)

            # make sentinel 2 figure
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            visualization.plot_prediction(ax, dataset, aoi_id, year, month)
            output_path = dataset_helpers.root_path() / 'plots' / 'workflow' / 'prediction'
            output_file = output_path / f'prediction_{aoi_id}_{year}_{month:02d}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)


def export_probability_cube(dataset: str, aoi_id: str):
    prob_cube = prediction_helpers.load_prediction_timeseries(dataset, aoi_id, dataset_helpers.include_masked())
    prob_cube = (prob_cube * 100).astype(np.uint8)
    transform, crs = dataset_helpers.get_geo(dataset, aoi_id)
    file = dataset_helpers.root_path() / 'plots' / 'workflow' / 'prob_cube' / f'prob_cube_{aoi_id}.tif'
    geofiles.write_tif(file, prob_cube, transform, crs)


def plot_constant_function_fit(dataset: str, aoi_id: str, pixel_coords: tuple):

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    probs_cube = prediction_helpers.load_prediction_timeseries(dataset, aoi_id, dataset_helpers.include_masked())
    m, n = pixel_coords
    ts = probs_cube[m, n, ]
    ax.scatter(np.arange(len(ts)), ts, c='k')

    mean = np.mean(ts)
    ax.plot((0, len(ts)-1), (mean, mean), label='$f_{c}(ts)$')

    y_ticks = np.linspace(0, 1, 6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.1f}' for y_tick in y_ticks], fontsize=16)
    ax.set_ylim((0, 1))
    ax.set_xlabel('$\it{time}$', fontsize=16)
    ax.set_xticks([])
    ax.set_ylabel('BUA probability', fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.legend(fontsize=16, loc='lower right', frameon=False)
    plt.show()


def plot_piecewise_constant_function_fit(dataset: str, aoi_id: str, pixel_coords: tuple, t_values: list):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fs = 18
    probs_cube = prediction_helpers.load_prediction_timeseries(dataset, aoi_id, dataset_helpers.include_masked())
    m, n = pixel_coords
    ts = probs_cube[m, n, ]
    ax.scatter(np.arange(len(ts)), ts, c='k')

    mean = np.mean(ts)
    ax.plot((0, len(ts)-1), (mean, mean), label='$f_{0}(x)$', c='k')

    cmap = visualization.DateColorMap(len(ts)).get_cmap()

    for t in t_values:
        mean_presegment = np.mean(ts[:t])
        mean_postsegment = np.mean(ts[t:])
        ts_fitted = t * [mean_presegment] + (len(ts)-t) * [mean_postsegment]
        x_values = np.arange(len(ts))

        ts_fitted = 2 * [mean_presegment] + 2 * [mean_postsegment]
        x_values = [0, t - 1 + 0.49, t - 1 + 0.51, len(ts) - 1]
        ax.plot(x_values, ts_fitted, label=f'$f_{{{t}}}(x)$', c=cmap(t))

    y_ticks = np.linspace(0, 1, 6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.1f}' for y_tick in y_ticks], fontsize=fs)
    ax.set_ylim((0, 1))
    ax.set_xlabel('$\it{time}$', fontsize=fs)
    ax.set_xticks([])
    ax.set_ylabel('BUA probability', fontsize=fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.legend(fontsize=fs, loc='lower right', frameon=False, ncol=2)
    plt.show()


def plot_change_detection_and_dating_results(dataset: str, aoi_id: str):
    model = cd_models.BreakPointDetection(error_multiplier=3, min_prob_diff=0.2, min_segment_length=2)
    change = model.change_detection(dataset, aoi_id, dataset_helpers.include_masked())
    change_date = model.change_dating(dataset, aoi_id, dataset_helpers.include_masked())

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    visualization.plot_blackwhite(ax, change)
    output_path = dataset_helpers.root_path() / 'plots' / 'workflow' / 'change'
    output_file = output_path / f'change_detection_{aoi_id}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    visualization.plot_change_date(ax, change_date,
                                   dataset_helpers.length_timeseries(dataset, aoi_id, dataset_helpers.include_masked()))
    output_path = dataset_helpers.root_path() / 'plots' / 'workflow' / 'change'
    output_file = output_path / f'change_dating_{aoi_id}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)


if __name__ == '__main__':
    ds = 'spacenet7'
    aoi_id = 'L15-0358E-1220N_1433_3310_13'
    pixel = (288, 206)
    # produce_workflow_data(ds, aoi_id)
    # export_probability_cube(ds, aoi_id)

    # plot_constant_function_fit(ds, aoi_id, pixel)
    # plot_piecewise_constant_function_fit(ds, aoi_id, pixel, [4, 10, 16])
    plot_change_detection_and_dating_results(ds, aoi_id)
