import numpy as np
from tqdm import tqdm
from utils import dataset_helpers, config, label_helpers, metrics, geofiles
import change_detection_models as cd_models
import matplotlib.pyplot as plt
FONTSIZE = 16

def quanitative_evaluation(dataset: str, model: cd_models.ChangeDetectionMethod) -> tuple:
    preds, gts = [], []
    for aoi_id in dataset_helpers.get_aoi_ids(dataset, min_timeseries_length=6):
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
    ax.set_ylabel('F1 score')
    plt.legend(fontsize=fontsize, frameon=False)
    plt.show()


# em: error multiplier, mdr: min diff probability
def run_grid_search(em_range: tuple, em_step_size: int, mdp_range: tuple, mdp_step_size: float):

    em_start, em_end = em_range
    em_candidates = np.arange(em_start, em_end + em_step_size, em_step_size)
    m = len(em_candidates)

    mdp_start, mdp_end = mdp_range
    mdp_candidates = np.arange(mdp_start, mdp_end + mdp_step_size, mdp_step_size)
    n = len(mdp_candidates)

    fname = f'grid_search_{config.input_sensor()}_{config.subset_activated("spacenet7")}.json'
    file = config.root_path() / 'grid_search' / fname

    if file.exists():
        ablation_data = geofiles.load_json(file)
    else:
        ablation_data = {
            'em_range': em_range,
            'em_step_size': em_step_size,
            'mdp_range': mdp_range,
            'mdp_step_size': mdp_step_size,
            'data': []
        }
        geofiles.write_json(file, ablation_data)

        for i, em_candidate in enumerate(em_candidates):
            for j, mdp_candidate in enumerate(tqdm(mdp_candidates)):
                sf = cd_models.StepFunctionModel(error_multiplier=em_candidate, min_prob_diff=mdp_candidate,
                                                 min_segment_length=2)
                f1, precision, recall = quanitative_evaluation('spacenet7', sf)
                ablation_data['data'].append({
                    'index': (i, j),
                    'em': int(em_candidate),
                    'mdp': float(mdp_candidate),
                    'f1_score': float(f1),
                    'precision': float(precision),
                    'recall': float(recall),
                })
        geofiles.write_json(file, ablation_data)

    f1_matrix = np.empty((m, n), dtype=np.single())
    prec_matrix = np.empty((m, n), dtype=np.single())
    rec_matrix = np.empty((m, n), dtype=np.single())

    for d in ablation_data['data']:
        i, j = d['index']
        f1_matrix[i, j] = d['f1_score']
        prec_matrix[i, j] = d['precision']
        rec_matrix[i, j] = d['recall']

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    vmax = 0.4
    img = ax.imshow(f1_matrix, vmin=0, vmax=vmax, cmap='jet')
    xticks = np.arange(0, n, 2)
    mdp_ticks = np.arange(mdp_start, mdp_end + mdp_step_size, mdp_step_size * 2)
    xticklabels = [f'{mdp_tick:.1f}' for mdp_tick in mdp_ticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=FONTSIZE)
    ax.set_xlabel(r'$\lambda_2$ (min probability increase)', fontsize=FONTSIZE)
    yticks = np.arange(m)
    yticklabels = [f'{cand:.0f}' for cand in em_candidates]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=FONTSIZE)
    ax.set_ylabel(r'$\lambda_1$ (error multiplier)', fontsize=FONTSIZE)
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel('F1 score', rotation=270, fontsize=FONTSIZE)
    cbartick_stepsize = 0.1
    cbarticks = np.arange(0, vmax + cbartick_stepsize, cbartick_stepsize)
    cbar_ticklabels = [f'{cbartick:.1f}' for cbartick in cbarticks]
    cbar.ax.get_yaxis().set_ticks(cbarticks)
    cbar.ax.get_yaxis().set_ticklabels(cbar_ticklabels, fontsize=FONTSIZE)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # ablation1(2, min_diff_range=(0, 1), step_size=0.05)
    # ablation2(0.4, error_multiplier_range=(1, 6))
    run_grid_search(em_range=(1, 6), em_step_size=1, mdp_range=(0, 1), mdp_step_size=0.05)

