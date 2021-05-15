from utils import dataset_helpers, geofiles
import numpy as np


def mask_index(dataset: str, aoi_id: str, year: int, month: int) -> int:
    md = dataset_helpers.metadata(dataset)['aois'][aoi_id]
    md_masked = [[y, m, mask, *_] for y, m, mask, *_ in md if mask]
    for i, (y, m, *_) in enumerate(md_masked):
        if year == y and month == m:
            return i


def has_mask(dataset: str, aoi_id: str, year: int, month: int) -> bool:
    md = dataset_helpers.metadata(dataset)['aois'][aoi_id]
    for y, m, mask, *_ in md:
        if year == y and m == month:
            return mask


def has_masked_timestamps(dataset: str, aoi_id: str) -> bool:
    masks_file = dataset_helpers.dataset_path(dataset) / aoi_id / f'masks_{aoi_id}.tif'
    


def load_mask(dataset: str, aoi_id: str, year: int, month: int) -> np.ndarray:
    if has_mask(dataset, aoi_id, year, month):
        index = mask_index(dataset, aoi_id, year, month)
        masks_file = dataset_helpers.dataset_path(dataset) / aoi_id / f'masks_{aoi_id}.tif'
        masks, *_ = geofiles.read_tif(masks_file)
        return (masks[:, :, index]).astype(np.bool)
    else:
        return np.zeros(shape=dataset_helpers.get_yx_size('spacenet7', aoi_id), dtype=np.bool)


if __name__ == '__main__':
    pass
