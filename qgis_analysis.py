from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from utils import config, dataset_helpers, input_helpers, label_helpers, geofiles
import change_detection_models as cd_models


def qgis_outputs(model: cd_models.ChangeDetectionMethod, aoi_id):
    scale_factor = 0.3
    dates = dataset_helpers.get_timeseries(aoi_id)
    start_year, start_month, *_ = dates[0]
    end_year, end_month, *_ = dates[-1]
    save_path = config.output_path() / 'qgis' / aoi_id
    save_path.mkdir(exist_ok=True)

    def visualize_s2(aoi_id, year, month):
        s2 = input_helpers.load_sentinel2(aoi_id, year, month)
        s2 = np.clip(s2[:, :, [2, 1, 0]] / scale_factor, 0, 1)
        s2[0, 0, :] = [0, 0, 0]
        s2[0, 1, :] = [1, 1, 1]
        fig, ax = plt.subplots(1, 1)
        ax.imshow(s2)
        ax.set_axis_off()
        s2_file = save_path / f's2_{aoi_id}_{year}_{month:2d}.png'
        plt.savefig(s2_file, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    visualize_s2(aoi_id, start_year, start_month)
    visualize_s2(aoi_id, end_year, end_month)

    def visualize_change(aoi_id, prediction):
        change = model.change_detection(aoi_id) if prediction else label_helpers.generate_change_label(aoi_id)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(change, cmap='gray', vmin=0, vmax=1)
        ax.set_axis_off()
        fname = f'pred_{aoi_id}.png' if prediction else f'label_{aoi_id}.png'
        change_file = save_path / fname
        plt.savefig(change_file, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    visualize_change(aoi_id, True)
    visualize_change(aoi_id, False)


if __name__ == '__main__':
    model = cd_models.SimpleStepFunctionModel()
    for aoi_id in dataset_helpers.get_aoi_ids(min_timeseries_length=config.min_timeseries_length()):
        qgis_outputs(model, aoi_id)