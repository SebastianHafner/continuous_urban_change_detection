from utils import geofiles, visualization, dataset_helpers, prediction_helpers, label_helpers, mask_helpers
import matplotlib.pyplot as plt
from tqdm import tqdm


def sanity_check_change_detection_label(dataset: str, aoi_id: str, include_masked_data: bool = False,
                                        include_buildings_label: bool = True, save_plot: bool = False):
    # buildings labels only exist for spacenet7 dataset
    include_buildings_label = include_buildings_label and dataset == 'spacenet7'
    dates = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)

    n_plots = 5 if include_buildings_label else 3
    fig, axs = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    # first
    year_first, month_first, *_ = dates[0]
    visualization.plot_optical(axs[0], dataset, aoi_id, year_first, month_first)
    if include_buildings_label:
        visualization.plot_buildings(axs[1], aoi_id, year_first, month_first)

    # last
    year_last, month_last, *_ = dates[-1]
    visualization.plot_optical(axs[2 if include_buildings_label else 1], dataset, aoi_id, year_last, month_last)
    if include_buildings_label:
        visualization.plot_buildings(axs[3], aoi_id, year_last, month_last)

    # change label
    visualization.plot_change_label(axs[4 if include_buildings_label else 2], dataset, aoi_id, include_masked_data)

    if not save_plot:
        plt.show()
    else:
        dataset_name = dataset_helpers.dataset_name(dataset)
        output_path = dataset_helpers.root_path() / 'plots' / 'sanity_check' / 'change_label' / dataset_name
        output_path.mkdir(exist_ok=True)
        output_file = output_path / f'{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def sanity_check_change_dating_label(dataset: str, aoi_id: str, include_masked_data: bool = False,
                                     save_plot: bool = False):
    ts = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)
    n_rows, n_cols = 1, len(ts) + 1
    plot_size = 3

    cmap = visualization.DateColorMap(len(ts)).get_cmap()

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * plot_size, n_rows * plot_size))

    for i, (year, month, *_) in enumerate(tqdm(ts)):
        visualization.plot_optical(axs[i], dataset, aoi_id, year, month)
        title = f'{year}-{month:02d}'
        color = cmap(i)
        axs[i].set_title(title, c=color, fontsize=16, fontweight='bold')

    visualization.plot_change_date_label(axs[-1], aoi_id, include_masked_data)

    if not save_plot:
        plt.show()
    else:
        output_path = dataset_helpers.root_path() / 'plots' / 'sanity_check' / 'date_label'
        output_path.parent.mkdir(exist_ok=True)
        output_file = output_path / f'{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def sanity_check_buildings_label(dataset: str, aoi_id: str, include_masked_data: bool = False, save_plot: bool = False):
    ts = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)
    n_rows, n_cols = 2, len(ts)
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * plot_size, n_rows * plot_size))

    for i, (year, month, *_) in enumerate(tqdm(ts)):
        visualization.plot_optical(axs[0, i], dataset, aoi_id, year, month)
        title = f'{i} {year}-{month:02d}'
        axs[0, i].set_title(title, c='k', fontsize=16, fontweight='bold')
        visualization.plot_buildings(axs[1, i], aoi_id, year, month)

    if not save_plot:
        plt.show()
    else:
        output_path = dataset_helpers.root_path() / 'plots' / 'sanity_check' / 'buildings_label'
        output_path.parent.mkdir(exist_ok=True)
        output_file = output_path / f'{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def sanity_check_masks(dataset: str, aoi_id, save_plot: bool = False):
    ts = dataset_helpers.get_timeseries(dataset, aoi_id, True)
    ts_masked = [[y, m, mask, *_] for y, m, mask, *_ in ts if mask]
    if len(ts_masked) < 5:
        # TODO make it work for len 1
        return

    n_rows, n_cols = 2, len(ts_masked)
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * plot_size, n_rows * plot_size))

    for i, (year, month, *_) in enumerate(tqdm(ts_masked)):
        visualization.plot_optical(axs[0, i], dataset, aoi_id, year, month)
        title = f'{year}-{month:02d}'
        axs[0, i].set_title(title, c='k', fontsize=16, fontweight='bold')
        visualization.plot_buildings(axs[1, i], aoi_id, year, month)

    if not save_plot:
        plt.show()
    else:
        output_path = dataset_helpers.root_path() / 'plots' / 'sanity_check' / 'masks'
        output_path.parent.mkdir(exist_ok=True)
        output_file = output_path / f'{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def sanity_check_mask_numbers(dataset: str, aoi_id: str):



if __name__ == '__main__':
    ds = 'spacenet7'
    for aoi_id in dataset_helpers.get_aoi_ids(ds):
        # sanity_check_change_detection_label(ds, aoi_id, include_masked_data=True, save_plot=False)
        # sanity_check_change_dating_label(ds, aoi_id, include_masked_data=True, save_plot=True)
        sanity_check_masks(ds, aoi_id, save_plot=False)
        pass

    # sanity_check_change_detection_label(ds, 'L15-0331E-1257N_1327_3160_13', include_masked_data=False, save_plot=False)
    # sanity_check_change_dating_label(ds, 'L15-1479E-1101N_5916_3785_13', include_masked_data=True, save_plot=False)
    # sanity_check_buildings_label(ds, 'L15-1479E-1101N_5916_3785_13', include_masked_data=True, save_plot=False)