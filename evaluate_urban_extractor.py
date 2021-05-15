import numpy as np
from utils import dataset_helpers, prediction_helpers, label_helpers, metrics
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_urban_extractor_evaluation(config_name: str, aoi_id: str, include_masked_data: bool = False):
    length_ts = dataset_helpers.length_timeseries('spacenet7', aoi_id)
    f1_scores, precisions, recalls = [], [], []

    for i in range(length_ts):
        label = label_helpers.load_label_in_timeseries(aoi_id, i)
        pred = prediction_helpers.load_prediction_in_timeseries(config_name, 'spacenet7_s1s2_dataset', aoi_id, i)
        pred = pred > 0.5
        f1_scores.append(metrics.compute_f1_score(pred, label))
        precisions.append(metrics.compute_precision(pred, label))
        recalls.append(metrics.compute_recall(pred, label))

    mean_f1, std_f1 = np.mean(f1_scores), np.std(f1_scores)
    mean_p, std_p = np.mean(precisions), np.std(precisions)
    mean_r, std_r = np.mean(recalls), np.std(recalls)

    print(aoi_id)
    print(f'F1 {mean_f1:.3f} ({std_f1:.3f}) - P {mean_p:.3f} ({std_p:.3f}) - R {mean_r:.3f} ({std_r:.3f})')


# https://stackoverflow.com/questions/22364565/python-pylab-scatter-plot-error-bars-the-error-on-each-point-is-unique
def show_precision_recall_evaluation(config_name: str):

    mean_f1, mean_p, mean_r = [], [], []
    std_f1, std_p, std_r = [], [], []

    aoi_ids = dataset_helpers.get_all_ids('spacenet7_s1s2_dataset')
    for aoi_id in tqdm(aoi_ids):
        length_ts = dataset_helpers.length_timeseries('spacenet7_s1s2_dataset', aoi_id)
        f1_scores, precisions, recalls = [], [], []

        for i in range(length_ts):
            label = label_helpers.load_label_in_timeseries(aoi_id, i)
            pred = prediction_helpers.load_prediction_in_timeseries(config_name, aoi_id, i)
            pred = pred > 0.5
            f1_scores.append(metrics.compute_f1_score(pred, label))
            precisions.append(metrics.compute_precision(pred, label))
            recalls.append(metrics.compute_recall(pred, label))

        mean_f1.append(np.mean(f1_scores))
        std_f1.append(np.std(f1_scores))
        mean_p.append(np.mean(precisions))
        std_p.append(np.std(precisions))
        mean_r.append(np.mean(recalls))
        std_r.append(np.std(recalls))

    fontsize = 20
    color = 'k'
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(mean_r, mean_p, c=color)
    ax.errorbar(mean_r, mean_p, xerr=std_r, yerr=std_p, linestyle="None", c=color)
    ticks = np.linspace(0, 1, 6)
    tick_labels = [f'{tick:0.1f}' for tick in ticks]
    ax.set_xlim((0, 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=fontsize)
    ax.set_xlabel('Recall', fontsize=fontsize)
    ax.set_ylabel('Precision', fontsize=fontsize)
    ax.set_ylim((0, 1))
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=fontsize)
    plt.show()


def show_f1_evaluation(include_masked_data: bool = False):

    data = []

    aoi_ids = dataset_helpers.get_aoi_ids('spacenet7')
    for aoi_id in tqdm(aoi_ids):

        length_ts = dataset_helpers.length_timeseries('spacenet7', aoi_id, include_masked_data)
        f1_scores, precisions, recalls = [], [], []

        for i in range(length_ts):
            label = label_helpers.load_label_in_timeseries(aoi_id, i, include_masked_data)
            pred = prediction_helpers.load_prediction_in_timeseries('spacenet7', aoi_id, i, include_masked_data)
            pred = pred > 0.5
            f1_scores.append(metrics.compute_f1_score(pred, label))

        data.append([aoi_id, f1_scores])

    data = sorted(data, key=lambda x: np.mean(x[1]))
    data_boxplots = [d[1] for d in data]

    fontsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(len(aoi_ids) / 2, 10))
    ax.boxplot(data_boxplots, whis=(0, 100))
    x_ticks = np.arange(len(data)) + 1
    x_tick_labels = [d[0][4:15] for d in data]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation=90, fontsize=fontsize)
    ax.set_ylabel('F1 score', fontsize=fontsize)
    y_ticks = np.linspace(0, 1, 6)
    ax.set_ylim((0, 1))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.1f}' for y_tick in y_ticks], fontsize=fontsize)
    plt.show()


if __name__ == '__main__':
    for aoi_id in dataset_helpers.get_aoi_ids('spacenet7'):
        # run_urban_extractor_evaluation(config_name, aoi_id)
        pass

    show_f1_evaluation(include_masked_data=True)
