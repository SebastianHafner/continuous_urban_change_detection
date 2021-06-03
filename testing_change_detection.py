from utils import dataset_helpers, label_helpers, visualization, metrics
import matplotlib.pyplot as plt
import change_detection_models as cd_models
import numpy as np
from tqdm import tqdm


def qualitative_testing(model: cd_models.ChangeDetectionMethod, dataset: str, aoi_id: str, save_plot: bool = False,
                        color_misclassifications: bool = False):

    dates = dataset_helpers.get_timeseries(dataset, aoi_id, dataset_helpers.include_masked())
    start_year, start_month, *_ = dates[0]
    end_year, end_month, *_ = dates[-1]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # pre image, post image and gt
    visualization.plot_optical(axs[0], dataset, aoi_id, start_year, start_month)
    axs[0].set_title('S2 Start TS')
    visualization.plot_optical(axs[1], dataset, aoi_id, end_year, end_month)
    axs[1].set_title('S2 End TS')

    visualization.plot_change_label(axs[2], dataset, aoi_id, dataset_helpers.include_masked())
    axs[2].set_title('Change GT')

    change = model.change_detection(dataset, aoi_id, dataset_helpers.include_masked())
    if color_misclassifications:
        visualization.plot_classification(axs[3], change, dataset, aoi_id, dataset_helpers.include_masked())
    else:
        visualization.plot_change_label(axs[3], dataset, aoi_id, dataset_helpers.include_masked())
    axs[3].set_title('Change Pred')

    if not save_plot:
        plt.show()
    else:
        save_path = dataset_helpers.root_path() / 'plots' / 'testing' / model.name / 'change_detection'
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'change_{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def quantitative_testing(model: cd_models.ChangeDetectionMethod, dataset: str, aoi_id: str):

    # TODO: different quantitative testing for oscd dataset (not penalizing omissions)
    pred = model.change_detection(dataset, aoi_id, dataset_helpers.include_masked())
    gt = label_helpers.generate_change_label(dataset, aoi_id, dataset_helpers.include_masked())

    precision = metrics.compute_precision(pred, gt)
    recall = metrics.compute_recall(pred, gt)
    f1 = metrics.compute_f1_score(pred, gt)

    print(aoi_id)
    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


def quantitative_testing_dataset(model: cd_models.ChangeDetectionMethod, dataset: str):
    preds, gts = [], []
    for aoi_id in tqdm(dataset_helpers.get_aoi_ids(dataset)):
        pred = model.change_detection(dataset, aoi_id, dataset_helpers.include_masked())
        preds.append(pred.flatten())
        gt = label_helpers.generate_change_label(dataset, aoi_id, dataset_helpers.include_masked())
        gts.append(gt.flatten())
        assert(pred.size == gt.size)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    precision = metrics.compute_precision(preds, gts)
    recall = metrics.compute_recall(preds, gts)
    f1 = metrics.compute_f1_score(preds, gts)

    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


if __name__ == '__main__':
    ds = 'spacenet7'

    dcva = cd_models.DeepChangeVectorAnalysis(subset_features=True)
    pcc = cd_models.PostClassificationComparison()
    thresholding = cd_models.Thresholding()
    sf = cd_models.StepFunctionModel(n_stable=6)
    bpd = cd_models.BreakPointDetection(error_multiplier=3, min_prob_diff=0.2, min_segment_length=2, noise_reduction=True)
    bbpd = cd_models.BackwardsBreakPointDetection(error_multiplier=2, min_prob_diff=0.1, min_segment_length=1,
                                                  improved_final_prediction=True)
    bpd = cd_models.BreakPointDetection(error_multiplier=2, min_prob_diff=0.2, min_segment_length=2)
    model = bpd
    for aoi_id in dataset_helpers.get_aoi_ids(ds):
        qualitative_testing(model, ds, aoi_id, save_plot=False, color_misclassifications=True)
        # quantitative_testing(model, ds, aoi_id)
        pass

    # qualitative_testing(model, ds, 'L15-0566E-1185N_2265_3451_13', save_plot=False)


    # quantitative_testing_dataset(model, ds)
    # quantitative_testing(model, ds, 'L15-0683E-1006N_2732_4164_13')

