from utils import dataset_helpers, input_helpers, label_helpers, mask_helpers, metrics, visualization, geofiles, config
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def timeseries_length_comparison_barcharts(dataset: str, numeric_names: bool = False, include_ts_duration: bool = False):
    labels, n_clear_sar, n_clear_fusion, duration = [], [], [], []
    aoi_ids = dataset_helpers.get_aoi_ids(dataset)
    for aoi_id in tqdm(aoi_ids):
        metadata = dataset_helpers.metadata(dataset)['aois'][aoi_id]
        n_clear_sar.append(len([_ for _, _, _, s1, s2 in metadata if s1]))
        n_clear_fusion.append(len([_ for _, _, _, s1, s2 in metadata if s1 and s2]))
        labels.append(aoi_id)
        d = dataset_helpers.duration_timeseries(dataset, aoi_id, config.include_masked())
        duration.append(d + 1)

    if numeric_names:
        labels = [f'{i + 1}' for i in range(len(labels))]
    else:
        labels = [aoi_id[4:15] for aoi_id in labels]
    width = 0.2
    inbetween_space = 0.1

    fontsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(len(aoi_ids), 6))

    center_pos = np.arange(len(aoi_ids))
    offset = (width + inbetween_space) / 2
    bar_sar = ax.bar(center_pos - offset, n_clear_sar, width, label='S1', color='#1f77b4')
    bar_fusion = ax.bar(center_pos + offset, n_clear_fusion, width, label='S1S2', color='#d62728')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=fontsize)
    ax.set_xlim((-0.5, len(labels) - 0.5))
    ax.set_xlabel('AOI', fontsize=fontsize)

    max_value = (max(n_clear_sar) // 5 + 1) * 5
    y_ticks = np.arange(0, max_value + 1, 5)
    ax.set_ylim((0, max_value))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.0f}' for y_tick in y_ticks], fontsize=fontsize)
    ax.set_ylabel('Time series length', fontsize=fontsize)

    ax.legend(ncol=2, handletextpad=0.4, columnspacing=1.2, frameon=False, loc='upper center', fontsize=fontsize)

    if include_ts_duration:
        for i, (rect_sar, rect_fusion) in enumerate(zip(bar_sar, bar_fusion)):
            height = max(rect_sar.get_height(), rect_fusion.get_height())
            plt.text(center_pos[i], height + 0.3, f'{duration[i]:.0f}', ha='center', va='bottom', fontsize=config.fontsize())

    plt.show()


def urban_extraction_comparison_boxplots(config_names: list, dataset: str, numeric_names: bool = False):

    width = 0.2
    inbetween_space = 0.1
    fontsize = 20
    colors = ['#1f77b4', '#d62728']
    labels = ['S1', 'S1S2']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    aoi_ids = dataset_helpers.get_aoi_ids(dataset)
    fig, ax = plt.subplots(1, 1, figsize=(len(aoi_ids), 6))

    for i, config_name in enumerate(config_names):
        data = []
        for aoi_id in tqdm(aoi_ids):
            metadata = dataset_helpers.metadata(dataset)['aois'][aoi_id]
            if i == 0:
                timeseries = [(year, month, mask, s1, s2) for year, month, mask, s1, s2 in metadata if s1]
            else:
                timeseries = [(year, month, mask, s1, s2) for year, month, mask, s1, s2 in metadata if s1 and s2]
            f1_scores = []

            for j, ts in enumerate(timeseries):
                year, month, mask, *_ = ts
                if mask_helpers.is_fully_masked(dataset, aoi_id, year, month):
                    continue
                label = label_helpers.load_label(aoi_id, year, month)
                pred = input_helpers.load_prediction_raw(dataset, aoi_id, year, month, config_name)
                pred = pred > 0.5
                f1_scores.append(metrics.compute_f1_score(pred, label))
            data.append(f1_scores)

        center_pos = np.arange(len(aoi_ids))
        offset = (width + inbetween_space) / 2
        pos = center_pos - offset if i == 0 else center_pos + offset
        pl = ax.boxplot(data, positions=pos, sym='', widths=width)
        set_box_color(pl, colors[i])

        # draw temporary red and blue lines and use them to create a legend
        ax.plot([], c=colors[i], label=labels[i])

    if numeric_names:
        x_tick_labels = [f'{i + 1}' for i in range(len(aoi_ids))]
    else:
        x_tick_labels = [aoi_id[4:15] for aoi_id in labels]

    ax.set_xticks(center_pos)
    ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=fontsize)
    ax.set_xlabel('AOI', fontsize=fontsize)
    ax.set_ylabel('F1 score', fontsize=fontsize)
    y_ticks = np.linspace(0, 1, 6)
    ax.set_ylim((0, 1))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.1f}' for y_tick in y_ticks], fontsize=fontsize)
    ax.legend(ncol=2, handletextpad=0.4, columnspacing=1.2, frameon=False, loc='upper center', fontsize=fontsize)
    plt.show()


def change_detection_comparison(model_name: str, config_names: list, dataset: str, aoi_id: str,
                                column_names: list = None, save_plot: bool = False):

    cols = 3 + len(config_names)
    fig, axs = plt.subplots(1, cols, figsize=(cols * config.plotsize(), config.plotsize()))

    start_year, start_month = dataset_helpers.get_date_from_index(0, dataset, aoi_id)
    visualization.plot_optical(axs[0], dataset, aoi_id, start_year, start_month, vis='true_color')

    end_year, end_month = dataset_helpers.get_date_from_index(-1, dataset, aoi_id)
    visualization.plot_optical(axs[1], dataset, aoi_id, end_year, end_month, vis='true_color')

    visualization.plot_change_label(axs[2], dataset, aoi_id)

    for i_cfg, config_name in enumerate(config_names):
        pred_file = config.root_path() / 'inference' / model_name / config_name / f'change_{aoi_id}.tif'
        pred, _, _ = geofiles.read_tif(pred_file)
        visualization.plot_blackwhite(axs[3 + i_cfg], pred)

    if column_names is not None:
        for j, col_name in enumerate(column_names):
            axs[j].set_xlabel(col_name, fontsize=config.fontsize())

    if save_plot:
        output_path = config.root_path() / 'plots' / 'input_comparison'
        output_path.mkdir(exist_ok=True)
        output_file = output_path / f'{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def change_detection_comparison_assembled(model_name: str, config_names: list, dataset: str, aoi_ids: list,
                                          column_names: list = None, row_names: list = None):

    rows = len(aoi_ids)
    cols = 3 + len(config_names)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * config.plotsize(), rows * config.plotsize()))

    for i, aoi_id in enumerate(tqdm(aoi_ids)):
        start_year, start_month = dataset_helpers.get_date_from_index(0, dataset, aoi_id)
        visualization.plot_optical(axs[i, 0], dataset, aoi_id, start_year, start_month, vis='true_color')

        end_year, end_month = dataset_helpers.get_date_from_index(-1, dataset, aoi_id)
        visualization.plot_optical(axs[i, 1], dataset, aoi_id, end_year, end_month, vis='true_color')

        visualization.plot_change_label(axs[i, 2], dataset, aoi_id)

        for i_cfg, config_name in enumerate(config_names):
            pred_file = config.root_path() / 'inference' / model_name / config_name / f'change_{aoi_id}.tif'
            pred, _, _ = geofiles.read_tif(pred_file)
            visualization.plot_blackwhite(axs[i, 3 + i_cfg], pred)

    if row_names is not None:
        for i, row_name in enumerate(row_names):
            axs[i, 0].set_ylabel(row_name, fontsize=config.fontsize())

    if column_names is not None:
        for j, col_name in enumerate(column_names):
            axs[-1, j].set_xlabel(col_name, fontsize=config.fontsize())

    plt.show()


def change_detection_comparison_assembled_v2(model_name: str, config_names: str, dataset: str, aoi_ids: list,
                                             column_names: list = None, row_names: list = None):
    fontsize = 20
    rows = len(aoi_ids)
    cols = 2 + len(config_names)

    plot_size = 3
    fig, axs = plt.subplots(rows, cols, figsize=(cols * plot_size, rows * plot_size))

    for i, aoi_id in enumerate(tqdm(aoi_ids)):
        start_year, start_month = dataset_helpers.get_date_from_index(0, dataset, aoi_id)
        visualization.plot_optical(axs[i, 0], dataset, aoi_id, start_year, start_month, vis='true_color')

        end_year, end_month = dataset_helpers.get_date_from_index(-1, dataset, aoi_id)
        visualization.plot_optical(axs[i, 1], dataset, aoi_id, end_year, end_month, vis='true_color')

        for i_cfg, config_name in enumerate(config_names):
            pred_file = config.root_path() / 'inference' / model_name / config_name / f'change_{aoi_id}.tif'
            pred, _, _ = geofiles.read_tif(pred_file)
            visualization.plot_classification(axs[i, 2 + i_cfg], pred, dataset, aoi_id)

    if row_names is not None:
        for i, row_name in enumerate(row_names):
            axs[i, 0].set_ylabel(row_name, fontsize=fontsize)

    if column_names is not None:
        for j, col_name in enumerate(column_names):
            axs[-1, j].set_xlabel(col_name, fontsize=fontsize)

    plt.show()


if __name__ == '__main__':
    ds = 'spacenet7'
    timeseries_length_comparison_barcharts(ds, numeric_names=True, include_ts_duration=True)
    config_names = ['sar_jaccardmorelikeloss', 'fusionda_cons05_jaccardmorelikeloss']
    # urban_extraction_comparison_boxplots(config_names, ds, numeric_names=True)
    aoi_ids = [
        'L15-0358E-1220N_1433_3310_13',
        'L15-0368E-1245N_1474_3210_13',
        'L15-0434E-1218N_1736_3318_13',
        'L15-0487E-1246N_1950_3207_13',
        'L15-0577E-1243N_2309_3217_13',
    ]
    column_names = [r'(a) S2 Img $t_1$', r'(b) S2 Img $t_n$', '(c) GT', '(d) S1', '(e) S1S2']
    row_names = ['AOI 3', 'AOI 5', 'AOI 7', 'AOI 8', 'AOI 12']
    # change_detection_comparison_assembled('stepfunction', config_names, ds, aoi_ids=aoi_ids, column_names=column_names,
    #                                       row_names=row_names)
    #
    # column_names = [r'(a) S2 Img $t_1$', r'(b) S2 Img $t_n$', '(c) S1', '(d) S1S2']
    # row_names = ['AOI 3', 'AOI 5', 'AOI 7', 'AOI 8', 'AOI 12']
    # change_detection_comparison_assembled_v2('stepfunction', config_names, ds, aoi_ids=aoi_ids,
    #                                          column_names=column_names, row_names=row_names)
    #
    # for aoi_id in tqdm(dataset_helpers.get_aoi_ids(ds)):
    #     change_detection_comparison('stepfunction', config_names, 'spacenet7', aoi_id, column_names=column_names,
    #                                 save_plot=True)