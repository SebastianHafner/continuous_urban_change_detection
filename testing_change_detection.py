from utils import dataset_helpers, label_helpers, visualization, metrics
import matplotlib.pyplot as plt
import change_detection_models as cd_models
import numpy as np
from tqdm import tqdm


def qualitative_testing(model: cd_models.ChangeDetectionMethod, dataset: str, aoi_id: str, save_plot: bool = False):

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


def quantitative_testing(model: cd_models.ChangeDetectionMethod, dataset: str, aoi_id: str):

    # TODO: different quantitative testing for oscd dataset (not penalizing omissions)
    pred = model.change_detection(dataset, aoi_id)
    gt = label_helpers.generate_change_label(dataset, aoi_id)

    precision = metrics.compute_precision(pred, gt)
    recall = metrics.compute_recall(pred, gt)
    f1 = metrics.compute_f1_score(pred, gt)

    print(aoi_id)
    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


def quantitative_testing_dataset(model: cd_models.ChangeDetectionMethod, dataset: str):
    preds, gts = [], []
    for aoi_id in tqdm(dataset_helpers.get_all_ids(dataset)):
        pred = model.change_detection(dataset, aoi_id)
        preds.append(pred)
        gt = label_helpers.generate_change_label(dataset, aoi_id)
        gts.append(gt)
        assert(pred.size() == gt.size())

    preds = np.stack(preds)
    gts = np.stack(gts)

    precision = metrics.compute_precision(preds, gts)
    recall = metrics.compute_recall(preds, gts)
    f1 = metrics.compute_f1_score(preds, gts)

    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


if __name__ == '__main__':
    cfg = 'fusionda_cons05_jaccardmorelikeloss'
    ds = 'spacenet7_s1s2_dataset'
    ds = 'oscd_multitemporal_dataset'

    dcva = cd_models.DeepChangeVectorAnalysis(cfg, subset_features=True)
    pcc = cd_models.PostClassificationComparison(cfg)
    thresholding = cd_models.Thresholding(cfg)
    stepfunction = cd_models.StepFunctionModel(cfg, n_stable=6)

    model = dcva
    for aoi_id in dataset_helpers.get_all_ids(ds):
        qualitative_testing(model, ds, aoi_id)
        # quantitative_testing(model, ds, aoi_id)
    # quantitative_testing_dataset(model, ds)
