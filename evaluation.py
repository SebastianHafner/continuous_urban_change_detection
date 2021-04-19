import numpy as np
import matplotlib.pyplot as plt
from utils import geofiles, dataset_helpers, label_helpers, prediction_helpers, metrics, visualization


def evaluate_change_detection(aoi_id: str, method: str):
    pred = prediction_helpers.load_prediction(aoi_id, method, 'change_detection')
    # TODO: use function from label_helpers for this
    label_file = dataset_helpers.dataset_path() / aoi_id / f'label_change_{aoi_id}.tif'
    label, _, _ = geofiles.read_tif(label_file)
    f1_score = metrics.compute_change_f1_score(pred.flatten(), label.flatten())
    precision = metrics.compute_change_precision(pred.flatten(), label.flatten())
    recall = metrics.compute_change_recall(pred.flatten(), label.flatten())
    print(f'f1 score {f1_score:.3f} - precision {precision:.3f} - recall {recall:.3f}')


def evaluate_change_dating(aoi_id: str, method: str):
    pass


def plot_evaluation_change_dating(aoi_id: str, method: str):
    label = label_helpers.load_endtoend_label(aoi_id)
    pred = prediction_helpers.load_prediction(aoi_id, method, 'change_dating')

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    visualization.plot_change_date(axs[0], label)
    visualization.plot_change_date(axs[1], pred)
    visualization.plot_change_data_bar(axs[2])
    plt.show()


if __name__ == '__main__':
    # evaluate_change_detection('L15-0331E-1257N_1327_3160_13', 'stepfunction')
    plot_evaluation_change_dating('L15-0331E-1257N_1327_3160_13', 'stepfunction')
