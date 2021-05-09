from utils import label_helpers, prediction_helpers, dataset_helpers, visualization, geofiles
import change_detection_models as cd_models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def qualitative_testing(model: cd_models.ChangeDatingMethod, aoi_id: str, save_plot: bool = False):

    dates = dataset_helpers.get_timeseries('spacenet7_s1s2_dataset', aoi_id)
    pred_change_date = model.change_dating('spacenet7_s1s2_dataset', aoi_id)

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


def quantitative_testing(model: cd_models.ChangeDatingMethod, aoi_id: str):
    label_change_date = label_helpers.generate_change_date_label(aoi_id)
    pred_change_date = model.change_dating('spacenet7_s1s2_dataset', aoi_id)

    correct_change = np.logical_and(label_change_date != 0, pred_change_date != 0)
    sdiff = np.square(label_change_date - pred_change_date)
    sdiff[np.logical_not(correct_change)] = np.NaN
    n = np.count_nonzero(~np.isnan(sdiff))
    rmse = np.sqrt(np.nansum(sdiff) / n)
    print(f'{aoi_id} RMSE: {rmse:.3f}')


def quantitative_testing_dataset(model: cd_models.ChangeDatingMethod):
    all_sdiff = []
    n_total = 0
    for aoi_id in tqdm(dataset_helpers.get_all_ids('spacenet7_s1s2_dataset')):
        label_change_date = label_helpers.generate_change_date_label(aoi_id)
        pred_change_date = model.change_dating('spacenet7_s1s2_dataset', aoi_id)

        correct_change = np.logical_and(label_change_date != 0, pred_change_date != 0)
        sdiff = np.square(label_change_date - pred_change_date)
        sdiff[np.logical_not(correct_change)] = np.NaN
        all_sdiff.append(sdiff.flatten())
        n = np.count_nonzero(~np.isnan(sdiff))
        n_total += n

    all_sdiff = np.concatenate(all_sdiff)
    rmse = np.sqrt(np.nansum(all_sdiff) / n_total)
    print(f'RMSE: {rmse:.3f}')


if __name__ == '__main__':

    cfg = 'fusionda_cons05_jaccardmorelikeloss'
    model = cd_models.StepFunctionModel(cfg, n_stable=6)
    # model = cd_models.ImprovedStepFunctionModel(cfg)
    # model = cd_models.ImprovedStepFunctionModelV2(cfg)

    aoi_ids = dataset_helpers.get_all_ids('spacenet7_s1s2_dataset')
    for i, aoi_id in enumerate(aoi_ids):
        print(i)
        # qualitative_testing(model, aoi_id, save_plot=False)
        quantitative_testing(model, aoi_id)
    # quantitative_testing_dataset(model)