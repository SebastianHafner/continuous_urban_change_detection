from tqdm import tqdm
import numpy as np
from utils import dataset_helpers, config, label_helpers, metrics, geofiles
import change_detection_models as cd_models
import matplotlib.pyplot as plt


def quanitative_evaluation(dataset: str, model: cd_models.ChangeDetectionMethod) -> tuple:
    preds, gts = [], []
    for aoi_id in dataset_helpers.get_aoi_ids(dataset):
        pred = model.change_detection(dataset, aoi_id)
        preds.append(pred.flatten())
        gt = label_helpers.generate_change_label(dataset, aoi_id, config.include_masked())
        gts.append(gt.flatten())
        assert (pred.size == gt.size)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    precision = metrics.compute_precision(preds, gts)
    recall = metrics.compute_recall(preds, gts)
    f1 = metrics.compute_f1_score(preds, gts)

    return f1, precision, recall


def ablation1(error_multiplier: int, min_diff_range: tuple, step_size: float, band: str = 'VV'):
    fontsize = 16
    min_diff_start, min_diff_end = min_diff_range
    min_diff_candidates = np.arange(min_diff_start, min_diff_end + step_size, step_size)
    file = config.root_path() / 'ablation' / f'ablation1_{config.input_sensor()}_{error_multiplier}.json'

    if file.exists():
        ablation_data = geofiles.load_json(file)
    else:
        ablation_data = {
            'min_diff_range': min_diff_range,
            'step_size': step_size,
            'f1_score': [],
            'precision': [],
            'recall': []
        }

        for min_diff_candidate in tqdm(min_diff_candidates):
            sf = cd_models.StepFunctionModel(error_multiplier=error_multiplier, min_prob_diff=min_diff_candidate,
                                             min_segment_length=2)
            f1, precision, recall = quanitative_evaluation('spacenet7', sf)
            ablation_data['f1_score'].append(f1)
            ablation_data['precision'].append(precision)
            ablation_data['recall'].append(recall)
        geofiles.write_json(file, ablation_data)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(min_diff_candidates, ablation_data['f1_score'], label='F1 score')
    ax.plot(min_diff_candidates, ablation_data['precision'], label='Precision')
    ax.plot(min_diff_candidates, ablation_data['recall'], label='Recall')

    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_xlim([min_diff_start, min_diff_end])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in ticks], fontsize=fontsize)
    ax.set_xlabel(r'$\lambda_2$ (min prob diff)', fontsize=fontsize)

    ax.set_ylim([0, 1])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in ticks], fontsize=fontsize)
    ax.set_ylabel('F1 score', fontsize=fontsize)
    plt.legend(fontsize=16, frameon=False)
    plt.show()


def ablation2(min_prob_diff: float, error_multiplier_range: tuple):
    fontsize = 16
    error_multiplier_start, error_multiplier_end = error_multiplier_range
    error_multiplier_candidates = np.arange(error_multiplier_start, error_multiplier_end + 1)
    file = config.root_path() / 'ablation' / f'ablation2_{config.input_sensor()}_{min_prob_diff*100:.0f}.json'

    if file.exists():
        ablation_data = geofiles.load_json(file)
    else:
        ablation_data = {
            'error_multiplier_range': error_multiplier_range,
            'f1_score': [],
            'precision': [],
            'recall': []
        }

        for error_multiplier_candidate in tqdm(error_multiplier_candidates):
            sf = cd_models.StepFunctionModel(error_multiplier=error_multiplier_candidate, min_prob_diff=min_prob_diff,
                                             min_segment_length=2)
            f1, precision, recall = quanitative_evaluation('spacenet7', sf)
            ablation_data['f1_score'].append(f1)
            ablation_data['precision'].append(precision)
            ablation_data['recall'].append(recall)
        geofiles.write_json(file, ablation_data)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(error_multiplier_candidates, ablation_data['f1_score'], label='F1 score')
    ax.plot(error_multiplier_candidates, ablation_data['precision'], label='Precision')
    ax.plot(error_multiplier_candidates, ablation_data['recall'], label='Recall')

    ax.set_xlim([error_multiplier_start, error_multiplier_end])
    ax.set_xticks(error_multiplier_candidates)
    ax.set_xticklabels([f'{tick:.0f}' for tick in error_multiplier_candidates], fontsize=16)
    ax.set_xlabel(r'$\lambda_1$ (error multiplier)', fontsize=fontsize)

    y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_ylim([0, 1])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=16)
    ax.set_ylabel('F1 score', fontsize=fontsize)
    plt.legend(fontsize=fontsize, frameon=False)
    plt.show()


if __name__ == '__main__':
    ablation1(2, min_diff_range=(0, 1), step_size=0.05)
    ablation2(0.4, error_multiplier_range=(1, 6))


