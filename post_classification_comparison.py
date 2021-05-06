from utils import geofiles, dataset_helpers, prediction_helpers, visualization
import matplotlib.pyplot as plt
import change_detection_models as cd_models
import numpy as np


def test_change_detection(model: cd_models.ChangeDetectionMethod, dataset: str, aoi_id: str,
                          save_plot: bool = False):

    dates = dataset_helpers.get_timeseries(dataset, aoi_id)
    start_date = dates[0][:-1]
    end_date = dates[-1][:-1]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # pre image, post image and gt
    visualization.plot_optical(axs[0], dataset, aoi_id, *start_date)
    axs[0].set_title('S2 Start TS')
    visualization.plot_optical(axs[1], dataset, aoi_id, *end_date)
    axs[1].set_title('S2 End TS')

    visualization.plot_change_label(axs[2], dataset, aoi_id)
    axs[2].set_title('Change GT')

    change = model.change_detection(dataset, aoi_id)
    visualization.plot_blackwhite(axs[3], change)
    axs[3].set_title('Change Pred')

    if not save_plot:
        plt.show()
    else:
        save_path = dataset_helpers.root_path() / 'plots' / 'testing' / model.name
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'change_{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    config_name = 'fusionda_cons05_jaccardmorelikeloss'
    ds = 'spacenet7_s1s2_dataset'
    model = cd_models.PostClassificationComparison(config_name)
    for aoi_id in dataset_helpers.load_aoi_selection():
        test_change_detection(model, ds, aoi_id)

