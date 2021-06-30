from utils import geofiles, dataset_helpers, mask_helpers
import numpy as np


def load_label(aoi_id: str, year: int, month: int) -> np.ndarray:
    buildings_path = dataset_helpers.dataset_path('spacenet7') / aoi_id / 'buildings'
    label_file = buildings_path / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
    label, _, _ = geofiles.read_tif(label_file)
    label = np.squeeze(label > 0).astype(np.float)
    mask = mask_helpers.load_mask('spacenet7', aoi_id, year, month)
    label = np.where(~mask, label, np.NaN)
    return label


def load_raw_label(aoi_id: str, year: int, month: int) -> np.ndarray:
    buildings_path = dataset_helpers.dataset_path('spacenet7') / aoi_id / 'buildings'
    label_file = buildings_path / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
    label, _, _ = geofiles.read_tif(label_file)
    label = np.squeeze(label).astype(np.float)
    mask = mask_helpers.load_mask('spacenet7', aoi_id, year, month)
    label = np.where(~mask, label, np.NaN)
    return label


def load_label_in_timeseries(aoi_id: str, index: int, include_masked_data: bool,
                             ignore_bad_data: bool = True) -> np.ndarray:
    dates = dataset_helpers.get_timeseries('spacenet7', aoi_id, include_masked_data, ignore_bad_data)
    year, month, *_ = dates[index]
    label = load_label(aoi_id, year, month)
    return label


def load_label_timeseries(aoi_id: str, include_masked_data: bool = False, ignore_bad_data: bool = False) -> np.ndarray:
    dates = dataset_helpers.get_timeseries('spacenet7', aoi_id, include_masked_data, ignore_bad_data)
    label_cube = np.zeros((*dataset_helpers.get_yx_size('spacenet7', aoi_id), len(dates)), dtype=np.float)
    for i, (year, month, *_) in enumerate(dates):
        label = load_label(aoi_id, year, month)
        label_cube[:, :, i] = label
    return label_cube


def generate_change_label(dataset: str, aoi_id: str, include_masked_data: bool = False,
                          ignore_bad_data: bool = True) -> np.ndarray:
    # computing it for spacenet7 (change between first and last label)
    if dataset == 'spacenet7':
        label_start = load_label_in_timeseries(aoi_id, 0, include_masked_data, ignore_bad_data)
        label_end = load_label_in_timeseries(aoi_id, -1, include_masked_data, ignore_bad_data)
        change = np.array(label_start != label_end)
    # for oscd the change label corresponds to the normal label
    else:
        label_file = dataset_helpers.dataset_path('oscd') / aoi_id / 'change' / f'change_{aoi_id}.tif'
        change, _, _ = geofiles.read_tif(label_file)
    return change.astype(np.uint8)


def generate_change_date_label(aoi_id: str, include_masked_data: bool = False,
                               ignore_bad_data: bool = True) -> np.ndarray:
    label_cube = load_label_timeseries(aoi_id, include_masked_data, ignore_bad_data)
    length_ts = dataset_helpers.length_timeseries('spacenet7', aoi_id, include_masked_data, ignore_bad_data)

    change_date_label = np.zeros((dataset_helpers.get_yx_size('spacenet7', aoi_id)), dtype=np.uint8)

    last_nonnan_label = label_cube[:, :, 0]
    for i in range(1, length_ts):

        prev_label, current_label = label_cube[:, :, i-1], label_cube[:, :, i]

        n_nan = np.sum(np.isnan(current_label))
        if n_nan > 0:
            debug = True

        # add immediate change (i.e., sequential labels are not NaN)
        immediate_valid = np.logical_and(~np.isnan(prev_label), ~np.isnan(current_label))
        immediate_change = np.logical_and(immediate_valid, np.array(prev_label != current_label))
        change_date_label[immediate_change] = i

        # check for change if previous is NaN using last nonnan label
        far_valid = np.logical_and(np.isnan(prev_label), ~np.isnan(current_label))
        far_change = np.logical_and(far_valid, np.array(last_nonnan_label != current_label))
        change_date_label[far_change] = i

        # change cannot be determined for this time step if current is NaN

        # update last nonnan label based on where current label is not NaN
        last_nonnan_label = np.where(~np.isnan(current_label), current_label, last_nonnan_label)

    return change_date_label


if __name__ == '__main__':
    a = np.array([True, True, False, np.NaN])
    b = np.array([True, np.NaN, False, np.NaN])
    print(np.logical_and(a, b))
    print(a == True)
    print(a > 0)
    pass
