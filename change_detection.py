from utils import dataset_helpers, label_helpers, visualization, metrics, geofiles, config
import matplotlib.pyplot as plt
import change_detection_models as cd_models
import numpy as np
from tqdm import tqdm
FONTSIZE = 16


def qualitative_testing(model: cd_models.ChangeDetectionMethod, aoi_id: str, save_plot: bool = False,
                        color_misclassifications: bool = False, sensor: str = 'sentinel2', show_f1: bool = True):

    dates = dataset_helpers.get_timeseries(aoi_id)
    start_year, start_month, *_ = dates[0]
    end_year, end_month, *_ = dates[-1]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # pre image, post image and gt
    if sensor == 'sentinel2':
        visualization.plot_optical(axs[0], aoi_id, start_year, start_month)
        visualization.plot_optical(axs[1], aoi_id, end_year, end_month)
    else:
        visualization.plot_sar(axs[0], aoi_id, start_year, start_month)
        visualization.plot_sar(axs[1], aoi_id, end_year, end_month)

    axs[0].set_title('S2 Start TS')
    axs[1].set_title('S2 End TS')

    visualization.plot_change_label(axs[2], aoi_id)
    axs[2].set_title('Change GT')

    change = model.change_detection(aoi_id)
    if color_misclassifications:
        visualization.plot_classification(axs[3], change,  aoi_id)
    else:
        visualization.plot_blackwhite(axs[3], change)
    title = 'Change Pred'
    if show_f1:
        label = label_helpers.generate_change_label(aoi_id)
        f1 = metrics.compute_f1_score(change.flatten(), label.flatten())
        title = f'{title} (F1 {f1:.3f})'
    axs[3].set_title(title)

    if not save_plot:
        plt.show()
    else:
        save_path = config.output_path() / 'plots' / 'change_detection'
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'change_{config.config_name()}_{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def quantitative_testing(model: cd_models.ChangeDetectionMethod, aoi_id: str):

    pred = model.change_detection(aoi_id)
    gt = label_helpers.generate_change_label(aoi_id)

    precision = metrics.compute_precision(pred, gt)
    recall = metrics.compute_recall(pred, gt)
    f1 = metrics.compute_f1_score(pred, gt)

    print(aoi_id)
    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


def quantitative_testing_dataset(model: cd_models.ChangeDetectionMethod):
    preds, gts = [], []
    for aoi_id in tqdm(dataset_helpers.get_aoi_ids(min_timeseries_length=config.min_timeseries_length())):
        pred = model.change_detection(aoi_id)
        preds.append(pred.flatten())
        gt = label_helpers.generate_change_label(aoi_id)
        gts.append(gt.flatten())
        assert(pred.size == gt.size)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    precision = metrics.compute_precision(preds, gts)
    recall = metrics.compute_recall(preds, gts)
    f1 = metrics.compute_f1_score(preds, gts)

    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


def run_change_detection_inference(model: cd_models.ChangeDetectionMethod):
    for aoi_id in tqdm(dataset_helpers.get_aoi_ids()):
        pred = model.change_detection(aoi_id)
        transform, crs = dataset_helpers.get_geo(aoi_id)
        path = config.root_path() / 'inference' / model.name / config.config_name()
        path.mkdir(exist_ok=True)
        file = path / f'change_{aoi_id}.tif'
        geofiles.write_tif(file, pred.astype(np.uint8), transform, crs)


def qualitative_testing_dataset(model: cd_models.ChangeDetectionMethod):
    f1, prec, rec = [], [], []
    for i, aoi_id in enumerate(tqdm(dataset_helpers.get_aoi_ids(min_timeseries_length=config.min_timeseries_length()))):
        pred = model.change_detection(aoi_id)
        gt = label_helpers.generate_change_label(aoi_id)
        prec.append(metrics.compute_precision(pred, gt))
        rec.append(metrics.compute_recall(pred, gt))
        f1.append(metrics.compute_f1_score(pred, gt))
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim((0, 1))
    yticks = np.linspace(0, 1, 5)
    yticklabels = [f'{ytick:.1f}' for ytick in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=FONTSIZE)
    ax.boxplot([f1, prec, rec], whis=[5, 95], notch=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['F1 score', 'Precision', 'Recall'], fontsize=FONTSIZE)
    plt.show()


if __name__ == '__main__':
    pcc = cd_models.PostClassificationComparison()
    thresholding = cd_models.Thresholding()
    sf = cd_models.StepFunctionModel(error_multiplier=2, min_prob_diff=0.3, min_segment_length=2)
    ksf = cd_models.KernelStepFunctionModel(kernel_size=3, error_multiplier=1, min_prob_diff=0.3, min_segment_length=2)
    sf = cd_models.SimpleStepFunctionModel()
    model = sf
    for i, aoi_id in enumerate((dataset_helpers.get_aoi_ids(min_timeseries_length=config.min_timeseries_length()))):
        # qualitative_testing(model, aoi_id, save_plot=True)
        # quantitative_testing(model, ds, aoi_id)
        pass

    quantitative_testing_dataset(model)
    # quantitative_testing(model, ds, 'L15-0683E-1006N_2732_4164_13')
    # run_change_detection_inference(sf, ds)
    # qualitative_testing_dataset(sf, ds)