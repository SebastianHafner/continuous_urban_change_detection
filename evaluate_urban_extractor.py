import numpy as np
from utils import dataset_helpers, prediction_helpers, label_helpers, metrics

def run_urban_extractor_evaluation(config_name: str, aoi_id: str):
    length_ts = dataset_helpers.length_time_series(aoi_id)
    f1_scores, precisions, recalls = [], [], []

    for i in range(length_ts):
        label = label_helpers.get_label_in_timeseries(aoi_id, i)
        pred = prediction_helpers.get_prediction_in_timeseries(config_name, aoi_id, i)
        pred = pred > 0.5
        f1_scores.append(metrics.compute_f1_score(pred, label))
        precisions.append(metrics.compute_precision(pred, label))
        recalls.append(metrics.compute_recall(pred, label))

    mean_f1, std_f1 = np.mean(f1_scores), np.std(f1_scores)
    mean_p, std_p = np.mean(precisions), np.std(precisions)
    mean_r, std_r = np.mean(recalls), np.std(recalls)

    print(aoi_id)
    print(f'F1 {mean_f1:.3f} ({std_f1:.3f}) - P {mean_p:.3f} ({std_p:.3f}) - R {mean_r:.3f} ({std_r:.3f})')



if __name__ == '__main__':
    config_name = 'fusionda_cons05_jaccardmorelikeloss'
    for aoi_id in dataset_helpers.load_aoi_selection():
        run_urban_extractor_evaluation(config_name, aoi_id)