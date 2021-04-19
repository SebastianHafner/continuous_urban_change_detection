import pandas as pd
from pathlib import Path
from utils import geofiles, dataset_helpers, label
from tqdm import tqdm

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
    if site_name == 'L15-1335E-1166N_5342_3524_13':
        print(mask_file.name)
        print(mask_file.exists())
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
    site_paths = [f for f in root_path.iterdir() if f.is_dir()]

    data = {
        's1_bands': ['VV', 'VH'],
        's2_bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B6', 'B8', 'B8A', 'B11', 'B12'],
        'sites': {}
    }

    for site_path in site_paths:

        s1_path = site_path / 'sentinel1'
        s1_files = [f for f in s1_path.glob('**/*')]

        months = [int(s1_file.stem.split('_')[-1]) for s1_file in s1_files]
        years = [int(s1_file.stem.split('_')[-2]) for s1_file in s1_files]
        dates = [(year, month) for year, month in zip(years, months)]
        dates = sorted(dates, key=lambda date: date[0] * 12 + date[1])

        data['sites'][site_path.name] = dates

    output_file = root_path / f'metadata.json'
    geofiles.write_json(output_file, data)


def generate_endtoend_labels():
    for aoi_id in tqdm(dataset_helpers.get_all_ids()):
        endtoend_label = label.generate_endtoend_label(aoi_id)
        geotransform, crs = dataset_helpers.get_geo(aoi_id)
        output_file = dataset_helpers.dataset_path() / aoi_id / f'{aoi_id}_endtoend_label.tif'
        geofiles.write_tif(output_file, endtoend_label, geotransform, crs)


if __name__ == '__main__':
    # assemble_buildings('train')
    # generate_dataset_file(ROOT_PATH / 'continuous_urban_change_detection' / 'spacenet7_s1s2_dataset')
    generate_endtoend_labels()
