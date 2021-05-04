from utils import label_helpers, prediction_helpers, dataset_helpers, visualization, geofiles
import change_detection_models as cd_models
from tqdm import tqdm
import matplotlib.pyplot as plt


def test_change_detection(model: cd_models.ChangeDetectionMethod, dataset: str, aoi_id: str, save_plot: bool = False):

    dates = dataset_helpers.get_time_series(dataset, aoi_id)
    start_date = dates[0][:-1]
    end_date = dates[-1][:-1]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # pre image, post image and gt
    visualization.plot_optical(axs[0, 0], dataset, aoi_id, *start_date)
    axs[0, 0].set_title('S2 Start TS')
    visualization.plot_optical(axs[0, 1], dataset, aoi_id, *end_date)
    axs[0, 1].set_title('S2 End TS')

    visualization.plot_change_label(axs[0, 2], dataset, aoi_id)
    axs[0, 2].set_title('GT')

    visualization.plot_prediction(axs[1, 0], model.config_name, dataset, aoi_id, *start_date)
    axs[1, 0].set_title('Pred Start TS')
    visualization.plot_prediction(axs[1, 1], model.config_name, dataset, aoi_id, *end_date)
    axs[1, 1].set_title('Pred End TS')

    change = model.change_detection(dataset, aoi_id)
    visualization.plot_blackwhite(axs[1, 2], change)

    if not save_plot:
        plt.show()
    else:
        save_path = dataset_helpers.root_path() / 'plots' / 'testing' / model.name
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'change_{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def test_change_dating(model: cd_models.ChangeDetectionMethod, dataset: str, aoi_id: str, save_plot: bool = False):
    # change detection
    change = model.change_detection(dataset, aoi_id)

    dates = dataset_helpers.get_time_series(aoi_id)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # pre image, post image and gt
    visualization.plot_optical(axs[0, 0], aoi_id, *dates[0])
    axs[0, 0].set_title('S2 Start TS')
    visualization.plot_optical(axs[0, 1], aoi_id, *dates[-1])
    axs[0, 1].set_title('S2 End TS')
    visualization.plot_change_label(axs[0, 2], aoi_id)
    axs[0, 2].set_title('GT')

    for i, model in enumerate(models):
        change = model.change_detection(aoi_id)
        visualization.plot_blackwhite(axs[1, i], change)
        if names is not None:
            axs[1, i].set_title(names[i])

    plt.show()

    change_date = model.change_dating(dataset, aoi_id)
    model_error = model.model_error(dataset, aoi_id)

    save_path = dataset_helpers.root_path() / 'inference' / 'advancedstepfunction'
    save_path.mkdir(exist_ok=True)

if __name__ == '__main__':

    ds = 'oscd_multitemporal_dataset'
    cfg = 'fusionda_cons05_jaccardmorelikeloss'
    # model = cd_models.BasicStepFunctionModel(cfg, n_stable=2)
    model = cd_models.ImprovedStepFunctionModel(cfg)

    aoi_ids = dataset_helpers.get_all_ids(ds)
    for aoi_id in tqdm(aoi_ids):
        test_change_detection(model, ds, aoi_id, save_plot=True)
