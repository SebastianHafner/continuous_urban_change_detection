import numpy as np
import matplotlib.pyplot as plt
from utils import geofiles, dataset_helpers, label_helpers, prediction_helpers, metrics, visualization
import change_detection_models as cd_models


def compare_change_detection_methods(aoi_id: str, models: list, names: list = None):

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


def qualitative_comparison_change_detection(dataset: str, aoi_id: str, models: list, names: list = None,
                                            save_plot: bool = False):

    dates = dataset_helpers.get_timeseries(dataset, aoi_id)
    start_date = dates[0][:-1]
    end_date = dates[-1][:-1]

    n_plots = 3 + len(models)
    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 5, 5))

    # pre image, post image and gt
    visualization.plot_optical(axs[0], dataset, aoi_id, *start_date)
    axs[0].set_title('S2 Start TS')
    visualization.plot_optical(axs[1], dataset, aoi_id, *end_date)
    axs[1].set_title('S2 End TS')

    visualization.plot_change_label(axs[2], dataset, aoi_id)
    axs[2].set_title('GT')

    for i, model in enumerate(models):
        pred_change = model.change_detection(dataset, aoi_id)
        ax_pred = fig.add_subplot(axs[3 + i])
        visualization.plot_blackwhite(ax_pred, pred_change)
        if names is not None:
            ax_pred.set_title(f'Pred {names[i]}', fontsize=20)
    if not save_plot:
        plt.show()
    else:
        save_path = dataset_helpers.root_path() / 'plots' / 'comparison' / 'change_detection'
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'change_{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def qualitative_comparison_change_dating(aoi_id: str, models: list, names: list = None, save_plot: bool = False):

    dates = dataset_helpers.get_timeseries('spacenet7_s1s2_dataset', aoi_id)

    ph = 20  # plot height
    n_plots = 3 + len(models)

    fig = plt.figure(figsize=(6 * n_plots, 7))
    grid = plt.GridSpec(ph+1, n_plots, wspace=0.1, hspace=0.1)

    ax_t1 = fig.add_subplot(grid[:ph, 0])
    visualization.plot_optical(ax_t1, 'spacenet7_s1s2_dataset', aoi_id, *dates[0][:-1])
    ax_t1.set_title('S2 Start TS', fontsize=20)
    ax_t2 = fig.add_subplot(grid[:ph, 1])
    visualization.plot_optical(ax_t2, 'spacenet7_s1s2_dataset', aoi_id, *dates[-1][:-1])
    ax_t2.set_title('S2 End TS', fontsize=20)

    ax_gt = fig.add_subplot(grid[:ph, 2])
    visualization.plot_change_date_label(ax_gt, aoi_id)
    ax_gt.set_title(f'GT', fontsize=20)

    ax_cbar = fig.add_subplot(grid[ph, :])
    visualization.plot_change_data_bar(ax_cbar, dates)

    for i, model in enumerate(models):
        pred_change_date = model.change_dating('spacenet7_s1s2_dataset', aoi_id)
        ax_pred = fig.add_subplot(grid[:ph, 3 + i])
        visualization.plot_change_date(ax_pred, pred_change_date, len(dates))
        if names is not None:
            ax_pred.set_title(f'Pred {names[i]}', fontsize=20)

    if not save_plot:
        plt.show()
    else:
        save_path = dataset_helpers.root_path() / 'plots' / 'comparison' / 'change_date'
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'{aoi_id}_change_date.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':

    cfg = 'fusionda_cons05_jaccardmorelikeloss'
    ds = 'spacenet7_s1s2_dataset'
    dcva = cd_models.DeepChangeVectorAnalysis(cfg)
    pcc = cd_models.PostClassificationComparison(cfg)
    thresh = cd_models.Thresholding(cfg)
    step = cd_models.StepFunctionModel(cfg, n_stable=2)
    step_refined = cd_models.StepFunctionModel(cfg, n_stable=6)
    step3 = cd_models.StepFunctionModel(cfg, n_stable=6, ts_extension=3)

    # model_comparison = [step, step2]
    # model_names = ['step', 'step refined']

    model_comparison = [pcc, thresh, step, step_refined]
    model_names = ['PCC', 'Thresh', 'Step', 'Step refined']

    # model_comparison = [dcva_model, step_model, adv_step_model]
    # model_names = ['DCVA (simplified)', 'Post-classification', 'Step function (refined)']

    for aoi_id in dataset_helpers.get_all_ids('spacenet7_s1s2_dataset'):
        print(aoi_id)
        # qualitative_comparison_change_dating(aoi_id, models=model_comparison, names=model_names, save_plot=True)
        qualitative_comparison_change_detection(ds, aoi_id, models=model_comparison, names=model_names, save_plot=True)