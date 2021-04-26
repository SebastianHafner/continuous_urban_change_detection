import numpy as np
import matplotlib.pyplot as plt
from utils import geofiles, dataset_helpers, label_helpers, prediction_helpers, metrics, visualization
import change_detection_models as cd_models


def compare_methods(aoi_id: str, models: list, names: list = None):

    dates = dataset_helpers.get_time_series(aoi_id)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # pre image, post image and gt
    visualization.plot_optical(axs[0, 0], aoi_id, *dates[0])
    axs[0, 0].set_title('S2 Start TS')
    visualization.plot_optical(axs[0, 1], aoi_id, *dates[-1])
    axs[0, 1].set_title('S2 End TS')
    visualization.plot_change_label(axs[0, 2], aoi_id)
    axs[0, 2].set_title('GT')

    for i, model in enumerate(models):
        change = model.change_detection(aoi_id)
        visualization.plot_blackwhite(axs[1, i], change)
        if names is not None:
            axs[1, i].set_title(names[i])

    plt.show()


if __name__ == '__main__':

    config_name = 'fusionda_cons05_jaccardmorelikeloss'
    dcva_model = cd_models.SimplifiedDeepChangeVectorAnalysis(config_name)
    pc_model = cd_models.PostClassificationComparison(config_name)
    step_model = cd_models.BasicStepFunctionModel(config_name)
    adv_step_model = cd_models.AdvancedStepFunctionModel(config_name)
    model_comparison = [dcva_model, step_model, adv_step_model]
    model_names = ['DCVA (simplified)', 'Post-classification', 'Step function (refined)']

    for aoi_id in dataset_helpers.load_aoi_selection():
        print(aoi_id)
        compare_methods(aoi_id, models=model_comparison, names=model_names)
