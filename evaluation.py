import numpy as np
import matplotlib.pyplot as plt
from utils import geofiles, dataset_helpers, label_helpers, prediction_helpers, metrics, visualization


def evaluate_change_detection(aoi_id: str, method: str):
    pred = prediction_helpers.load_prediction(aoi_id, method, 'change_dating')
    label_file = dataset_helpers.dataset_path() / aoi_id / f'label_endtoend_{aoi_id}.tif'
    label, _, _ = geofiles.read_tif(label_file)

    # background
    y = label == 0
    y_hat = pred == 0
    f1_score = metrics.compute_f1_score(y_hat.flatten(), y.flatten())
    precision = metrics.compute_precision(y_hat.flatten(), y.flatten())
    recall = metrics.compute_recall(y_hat.flatten(), y.flatten())
    print(f'BG: f1 score {f1_score:.3f} - precision {precision:.3f} - recall {recall:.3f}')

    # BUA
    y = label == 1
    y_hat = pred == 1
    f1_score = metrics.compute_f1_score(y_hat.flatten(), y.flatten())
    precision = metrics.compute_precision(y_hat.flatten(), y.flatten())
    recall = metrics.compute_recall(y_hat.flatten(), y.flatten())
    print(f'BUA: f1 score {f1_score:.3f} - precision {precision:.3f} - recall {recall:.3f}')

    # change
    y = label > 1
    y_hat = pred > 1
    f1_score = metrics.compute_f1_score(y_hat.flatten(), y.flatten())
    precision = metrics.compute_precision(y_hat.flatten(), y.flatten())
    recall = metrics.compute_recall(y_hat.flatten(), y.flatten())
    print(f'Change: f1 score {f1_score:.3f} - precision {precision:.3f} - recall {recall:.3f}')



def plot_change_detection_results(aoi_id: str, method: str):
    dates = dataset_helpers.get_time_series(aoi_id)
    change_date_label = label_helpers.generate_change_date_label(aoi_id)
    change_date_pred = prediction_helpers.load_prediction(aoi_id, method, 'change_dating')

    fig = plt.figure(figsize=(24, 8))
    grid = plt.GridSpec(15, 3, wspace=0.2, hspace=0.5)
    ax_gt = fig.add_subplot(grid[:-1, 0])
    ax_gt.set_title('GT', fontsize=20)
    ax_pred = fig.add_subplot(grid[:-1, 1])
    ax_pred.set_title(f'Pred {method}', fontsize=20)
    ax_error = fig.add_subplot(grid[:-1, 2])
    ax_error.set_title('Model error', fontsize=20)
    cbar_date = fig.add_subplot(grid[-1, :2])
    cbar_error = fig.add_subplot(grid[-1, 2])
    visualization.plot_change_date(ax_gt, change_date_label, len(dates))
    visualization.plot_change_date(ax_pred, change_date_pred, len(dates))
    visualization.plot_change_data_bar(cbar_date, dates)
    visualization.plot_model_error(ax_error, method, aoi_id)
    visualization.plot_model_error_bar(cbar_error)
    plt.show()


def plot_change_dating_results(aoi_id: str, method: str):
    dates = dataset_helpers.get_time_series(aoi_id)
    change_date_label = label_helpers.generate_change_date_label(aoi_id)
    change_date_pred = prediction_helpers.load_prediction(aoi_id, method, 'change_dating')

    fig = plt.figure(figsize=(24, 8))
    grid = plt.GridSpec(15, 3, wspace=0.2, hspace=0.5)
    ax_gt = fig.add_subplot(grid[:-1, 0])
    ax_gt.set_title('GT', fontsize=20)
    ax_pred = fig.add_subplot(grid[:-1, 1])
    ax_pred.set_title(f'Pred {method}', fontsize=20)
    ax_error = fig.add_subplot(grid[:-1, 2])
    ax_error.set_title('Model error', fontsize=20)
    cbar_date = fig.add_subplot(grid[-1, :2])
    cbar_error = fig.add_subplot(grid[-1, 2])
    visualization.plot_change_date(ax_gt, change_date_label, len(dates))
    visualization.plot_change_date(ax_pred, change_date_pred, len(dates))
    visualization.plot_change_data_bar(cbar_date, dates)
    visualization.plot_model_error(ax_error, method, aoi_id)
    visualization.plot_model_error_bar(cbar_error)
    plt.show()


if __name__ == '__main__':
    # evaluate_change_detection('L15-0331E-1257N_1327_3160_13', 'stepfunction')
    aoi_ids = dataset_helpers.load_aoi_selection()
    for aoi_id in aoi_ids:
        print(aoi_id)
        plot_change_dating_results(aoi_id, 'stepfunction')
        # evaluate_change_detection(aoi_id, 'stepfunction')