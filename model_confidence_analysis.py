from utils import dataset_helpers
import change_detection_models as cd_models
from tqdm import tqdm

def qualitative_analysis_change_detection(model: cd_models.StepFunctionModel, dataset: str,
                                          include_masked_date: bool = False):

    y_true, y_pred, confidence = [], [], []
    for aoi_id in tqdm(dataset_helpers.get_aoi_ids(dataset)):
        change = model.change_detection(dataset, aoi_id, include_masked_date)
    pass


if __name__ == '__main__':
    pass