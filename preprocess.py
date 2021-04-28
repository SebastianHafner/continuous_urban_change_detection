import pandas as pd
from pathlib import Path
from utils import geofiles, dataset_helpers, label_helpers
from tqdm import tqdm
import numpy as np

ROOT_PATH = Path('/storage/shafner')


# helper function to get date from label file name
# global_monthly_2020_01_mosaic_L15-1335E-1166N_5342_3524_13_Buildings.geojson
def get_date(label_file: Path) -> tuple:
    file_name = label_file.stem
    file_parts = file_name.split('_')
    year, month = int(file_parts[2]), int(file_parts[3])
    return year, month


# helper function to check if mask exists for time stamp
def has_mask(dataset: str, site_name: str, year: int, month: int) -> bool:
    mask_path = ROOT_PATH / 'spacenet7' / dataset / site_name / 'UDM_masks'
    mask_file = mask_path / f'global_monthly_{year}_{month:02d}_mosaic_{site_name}_UDM.tif'
    return mask_file.exists()


# creates data frame with columns: aoi_id; year; month; mask
def assemble_metadata(dataset: str):
    sn7_path = ROOT_PATH / 'spacenet7'
    dataset_path = sn7_path / dataset
    data = {'aoi_id': [], 'year': [], 'month': [], 'mask': []}

    site_paths = [f for f in dataset_path.iterdir() if f.is_dir()]

    def process_site(site_path: Path):
        site_name = site_path.name
        labels_path = site_path / 'labels_match'
        label_files = [f for f in labels_path.glob('**/*')]

        dates = [get_date(label_file) for label_file in label_files]
        # sort the dates
        dates = sorted(dates, key=lambda date: date[0] * 12 + date[1])

        for year, month in dates:
            data['aoi_id'].append(site_name)
            data['year'].append(year)
            data['month'].append(month)
            data['mask'].append(has_mask(dataset, site_name, year, month))

    for site_path in site_paths:
        process_site(site_path)

    df = pd.DataFrame.from_dict(data)
    print(df.shape)
    output_file = sn7_path / 'sn7_timestamps.csv'
    df.to_csv(output_file)


# merges buildings from a time series for a site into one geojson file
def assemble_buildings(dataset: str):
    sn7_path = ROOT_PATH / 'spacenet7'
    dataset_path = sn7_path / dataset
    site_paths = [f for f in dataset_path.iterdir() if f.is_dir()]

    for site_path in site_paths:
        site = site_path.name
        print(f'processing site: {site}')
        all_buildings = None

        labels_path = site_path / 'labels_match'
        label_files = [f for f in labels_path.glob('**/*')]

        for label_file in tqdm(label_files):
            data = geofiles.load_json(label_file)
            if all_buildings is None:
                all_buildings = dict(data)
                all_buildings['features'] = []

            year, month = get_date(label_file)
            features = []
            for f in data['features']:
                p = f['properties']
                properties = {'Id': p['Id'], 'year': year, 'month': month}
                f['properties'] = properties
                features.append(f)

            all_buildings['features'].extend(features)

        output_file = sn7_path / 'buildings_assembled' / f'buildings_{site}.geojson'
        geofiles.write_json(output_file, all_buildings)


def generate_dataset_file(path_to_spacenet7_s1s2_dataset: Path):
    root_path = path_to_spacenet7_s1s2_dataset
    timestamps_file = ROOT_PATH / 'spacenet7' / 'sn7_timestamps.csv'
    df = pd.read_csv(timestamps_file)

    missing_aois = geofiles.load_json(Path('missing_aois.json'))

    data = {
        's1_bands': ['VV', 'VH'],
        's2_bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B6', 'B8', 'B8A', 'B11', 'B12'],
        'sites': {}
    }

    for index, row in df.iterrows():
        aoi_id = row['aoi_id']
        aoi_data = [int(row['year']), int(row['month']), row['mask']]

        if aoi_id in missing_aois:
            continue

        if aoi_id not in data['sites'].keys():
            data['sites'][aoi_id] = []
        data['sites'][aoi_id].append(aoi_data)

    for aoi_id in data['sites'].keys():
        aoi_data = data['sites'][aoi_id]
        aoi_data = sorted(aoi_data, key=lambda date: date[0] * 12 + date[1])
        data['sites'][aoi_id] = aoi_data

    output_file = root_path / f'metadata.json'
    geofiles.write_json(output_file, data)


if __name__ == '__main__':
    # assemble_buildings('train')
    generate_dataset_file(ROOT_PATH / 'continuous_urban_change_detection' / 'spacenet7_s1s2_dataset')

