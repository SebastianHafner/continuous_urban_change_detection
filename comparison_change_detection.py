import numpy as np
import matplotlib.pyplot as plt
from utils import geofiles, dataset_helpers, label_helpers, prediction_helpers, metrics, visualization
import change_detection_models as cd_models


def qualitative_comparison_change_detection(models: list, dataset: str, aoi_id: str, names: list = None):

    dates = dataset_helpers.get_timeseries(ds, aoi_id)

    n_models = len(models)
    fig, axs = plt.subplots(1, 3 + n_models, figsize=((3 + n_models) * 5, 5))

    # pre image, post image and gt
    visualization.plot_optical(axs[0], dataset, aoi_id, *dates[0][:-1])
    axs[0].set_title('S2 Start TS')
    visualization.plot_optical(axs[1], dataset, aoi_id, *dates[-1][:-1])
    axs[1].set_title('S2 End TS')
    visualization.plot_change_label(axs[2], dataset, aoi_id)
    axs[2].set_title('Change GT')

    for i, model in enumerate(models):
        change = model.change_detection(dataset, aoi_id)
        visualization.plot_blackwhite(axs[3 + i], change)
        title = names[i] if names is not None else model.name
        axs[3 + i].set_title(title)

    plt.show()


if __name__ == '__main__':

    cfg = 'fusionda_cons05_jaccardmorelikeloss'
    ds = 'spacenet7_s1s2_dataset'

    dcva = cd_models.SimplifiedDeepChangeVectorAnalysis(cfg)
    pcc = cd_models.PostClassificationComparison(cfg)
    stepfunction = cd_models.StepFunctionModel(cfg, n_stable=6)

    model_comparison = [pcc, stepfunction]

    for aoi_id in dataset_helpers.get_all_ids(ds):
        qualitative_comparison_change_detection(model_comparison, ds, aoi_id)
