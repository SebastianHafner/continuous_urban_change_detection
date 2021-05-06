from utils import dataset_helpers, label_helpers, prediction_helpers, visualization, metrics
import matplotlib.pyplot as plt
import change_detection_models as cd_models
import numpy as np
from tqdm import tqdm

FONTSIZE = 20

# supported modes: first_last, all
def sensitivity_testing(model: cd_models.ChangeDetectionMethod, mode: str, save_plot: bool = False):

    f1_scores_extraction = []
    f1_scores_change = []

    for i, aoi_id in enumerate(tqdm(dataset_helpers.get_all_ids('spacenet7_s1s2_dataset'))):

        # compute average urban extraction f1 score
        f1_scores_ts = []
        ts = dataset_helpers.get_timeseries('spacenet7_s1s2_dataset', aoi_id)
        if mode == 'first_last':
            ts = [ts[0], ts[-1]]
        for year, month, _ in ts:
            label_bua = label_helpers.load_label(aoi_id, year, month)
            label_bua = label_bua > 0.5
            pred_bua = prediction_helpers.load_prediction(model.config_name, 'spacenet7_s1s2_dataset', aoi_id, year, month)
            pred_bua = pred_bua > 0.5
            f1_scores_ts.append(metrics.compute_f1_score(pred_bua, label_bua))
        f1_scores_extraction.append(np.mean(f1_scores_ts))

        # compute change f1 score
        pred_change = model.change_detection('spacenet7_s1s2_dataset', aoi_id)
        label_change = label_helpers.generate_change_label('spacenet7_s1s2_dataset', aoi_id)
        f1_scores_change.append(metrics.compute_f1_score(pred_change, label_change))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(f1_scores_extraction, f1_scores_change, c='k')
    ticks = np.linspace(0, 1, 6)
    tick_labels = [f'{tick:.1f}' for tick in ticks]
    ax.set_xlim((0, 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=FONTSIZE)
    ax.set_xlabel('F1 score (urban extraction)', fontsize=FONTSIZE)
    ax.set_ylim((0, 1))
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=FONTSIZE)
    ax.set_ylabel('F1 score (change detection)', fontsize=FONTSIZE)

    if not save_plot:
        plt.show()
    else:
        save_path = dataset_helpers.root_path() / 'plots' / 'sensitivity_analysis'
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'{model.name}_{mode}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    cfg = 'fusionda_cons05_jaccardmorelikeloss'
    ds = 'spacenet7_s1s2_dataset'

    pcc = cd_models.PostClassificationComparison(cfg)
    thresholding = cd_models.Thresholding(cfg)
    stepfunction = cd_models.StepFunctionModel(cfg, n_stable=6)

    sensitivity_testing(pcc, 'first_last', save_plot=True)
