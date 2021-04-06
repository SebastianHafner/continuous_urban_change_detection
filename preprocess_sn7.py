import pandas as pd
from pathlib import Path



ROOT_PATH = Path('/storage/shafner')



def assemble_metadata(dataset: str):
    sn7_path = ROOT_PATH / 'spacenet7' / dataset

    site_paths = [f for f in sn7_path.iterdir() if f.is_dir()]

    def process_site(site_path: Path):
        site_name = site_path.name
        labels_path = site_path / 'labels_match'
        label_files = [f for f in labels_path.glob('**/*')]

        def get_date(label_file: Path):
            file_name = label_file.stem
            file_parts = file_name.split('_')
            year, month = file_parts[2], file_parts[3]
            return f'{year}_{month}'

        dates = [get_date(label_file) for label_file in label_files]
        for 
        print(len(label_files))

    for site_path in site_paths:
        process_site(site_path)



if __name__ == '__main__':
    assemble_metadata('train')
    pass