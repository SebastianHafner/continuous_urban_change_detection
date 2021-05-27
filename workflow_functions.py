from pathlib import Path
from utils import geofiles, visualization, dataset_helpers, prediction_helpers, label_helpers, metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


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


if __name__ == '__main__':
    ds = 'spacenet7'
    aoi_id = 'L15-0358E-1220N_1433_3310_13'
    # produce_workflow_data(ds, aoi_id)
    export_probability_cube(ds, aoi_id)