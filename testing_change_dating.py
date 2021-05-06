from utils import label_helpers, prediction_helpers, dataset_helpers, visualization, geofiles
import change_detection_models as cd_models
from tqdm import tqdm
import matplotlib.pyplot as plt


def qualitative_testing(model: cd_models.StepFunctionModel, aoi_id: str, save_plot: bool = False):

    dates = dataset_helpers.get_timeseries('spacenet7_s1s2_dataset', aoi_id)
    pred_change_date = model.change_dating('spacenet7_s1s2_dataset', aoi_id)

    # fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    #
    # # pre image, post image and gt
    # visualization.plot_optical(axs[0], 'spacenet7_s1s2_dataset', aoi_id, *dates[0][:-1])
    # axs[0].set_title('S2 Start TS')
    # visualization.plot_optical(axs[1], 'spacenet7_s1s2_dataset', aoi_id, *dates[-1][:-1])
    # axs[1].set_title('S2 End TS')
    # visualization.plot_change_date_label(axs[2], aoi_id)
    # axs[2].set_title('Change Timestamp GT')
    #
    # visualization.plot_change_date(axs[3], pred_change_date, len(dates))
    # axs[3].set_title('Change Timestamp Pred')

    fig = plt.figure(figsize=(25, 7))
    ph = 20  # plot height
    grid = plt.GridSpec(ph+1, 4, wspace=0.1, hspace=0.1)
    ax_t1 = fig.add_subplot(grid[:ph, 0])
    visualization.plot_optical(ax_t1, 'spacenet7_s1s2_dataset', aoi_id, *dates[0][:-1])
    ax_t1.set_title('S2 Start TS', fontsize=20)
    ax_t2 = fig.add_subplot(grid[:ph, 1])
    visualization.plot_optical(ax_t2, 'spacenet7_s1s2_dataset', aoi_id, *dates[-1][:-1])
    ax_t2.set_title('S2 End TS', fontsize=20)

    ax_gt = fig.add_subplot(grid[:ph, 2])
    visualization.plot_change_date_label(ax_gt, aoi_id)
    ax_gt.set_title(f'Change Timestamp GT', fontsize=20)

    ax_pred = fig.add_subplot(grid[:ph, 3])
    visualization.plot_change_date(ax_pred, pred_change_date, len(dates))
    ax_pred.set_title(f'Change Timestamp Pred', fontsize=20)

    ax_cbar = fig.add_subplot(grid[ph, :])
    visualization.plot_change_data_bar(ax_cbar, dates)

    if not save_plot:
        plt.show()
    else:
        save_path = dataset_helpers.root_path() / 'plots' / 'testing' / model.name / 'change_date'
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'{aoi_id}_change_date.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def quantitative_testing(model: cd_models.StepFunctionModel, aoi_id: str, save_plot: bool = False):
    pass


if __name__ == '__main__':

    cfg = 'fusionda_cons05_jaccardmorelikeloss'
    model = cd_models.StepFunctionModel(cfg, n_stable=6)
    # model = cd_models.ImprovedStepFunctionModel(cfg)
    # model = cd_models.ImprovedStepFunctionModelV2(cfg)

    aoi_ids = dataset_helpers.get_all_ids('spacenet7_s1s2_dataset')
    for aoi_id in tqdm(aoi_ids):
        qualitative_testing(model, aoi_id, save_plot=True)