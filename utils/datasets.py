import torch
from torchvision import transforms

import json

from pathlib import Path
from abc import abstractmethod
from utils.augmentations import *
from utils.geotiff import *
import affine


class AbstractUrbanExtractionDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.DATASETS.PATH)

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        s1_bands = ['VV', 'VH']
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B6', 'B8', 'B8A', 'B11', 'B12']
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _get_sentinel1_data(self, site, patch_id):
        file = self.root_path / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, site, patch_id):
        file = self.root_path / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_auxiliary_data(self, aux_input, site, patch_id):
        file = self.root_path / site / aux_input / f'{aux_input}_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, site, patch_id):
        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH

        label_file = self.root_path / site / label / f'{label}_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(label_file)
        if threshold >= 0:
            img = img > threshold

        return np.nan_to_num(img).astype(np.float32), transform, crs

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]


# dataset for urban extraction with building footprints
class UrbanExtractionDataset(AbstractUrbanExtractionDataset):

    def __init__(self, cfg, dataset: str, include_projection: bool = False, no_augmentations: bool = False,
                 include_unlabeled: bool = True):
        super().__init__(cfg)

        self.dataset = dataset
        if dataset == 'training':
            self.sites = list(cfg.DATASETS.TRAINING)
            # using parameter include_unlabeled to overwrite config
            if include_unlabeled and cfg.DATALOADER.INCLUDE_UNLABELED:
                self.sites += cfg.DATASETS.UNLABELED
        elif dataset == 'validation':
            self.sites = list(cfg.DATASETS.VALIDATION)
        else:  # used to load only 1 city passed as dataset
            self.sites = [dataset]

        self.no_augmentations = no_augmentations
        self.transform = transforms.Compose([Numpy2Torch()]) if no_augmentations else compose_transformations(cfg)

        self.samples = []
        for site in self.sites:
            samples_file = self.root_path / site / 'samples.json'
            metadata = load_json(samples_file)
            samples = metadata['samples']
            # making sure unlabeled data is not used as labeled when labels exist
            if include_unlabeled and site in cfg.DATASETS.UNLABELED:
                for sample in samples:
                    sample['is_labeled'] = False
            self.samples += samples
        self.length = len(self.samples)
        self.n_labeled = len([s for s in self.samples if s['is_labeled']])

        self.include_projection = include_projection

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]

        site = sample['site']
        patch_id = sample['patch_id']
        is_labeled = sample['is_labeled']

        # loading images
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, geotransform, crs = self._get_sentinel2_data(site, patch_id)
        elif mode == 'sar':
            img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
        else:  # fusion baby!!!
            s1_img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
            s2_img, _, _ = self._get_sentinel2_data(site, patch_id)
            img = np.concatenate([s1_img, s2_img], axis=-1)

        # here we load all auxiliary inputs and append them to features
        aux_inputs = self.cfg.DATALOADER.AUXILIARY_INPUTS
        for aux_input in aux_inputs:
            aux_img, _, _ = self._get_auxiliary_data(aux_input, site, patch_id)
            img = np.concatenate([aux_img, img], axis=-1)

        # create dummy label if unlabeled
        if is_labeled:
            label, _, _ = self._get_label_data(site, patch_id)
        else:
            label = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)

        img, label = self.transform((img, label))

        item = {
            'x': img,
            'y': label,
            'site': site,
            'patch_id': patch_id,
            'is_labeled': sample['is_labeled'],
            'image_weight': np.float(sample['img_weight']),
        }

        if self.include_projection:
            item['transform'] = geotransform
            item['crs'] = crs

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        labeled_perc = self.n_labeled / self.length * 100
        return f'Dataset with {self.length} samples ({labeled_perc:.1f} % labeled) across {len(self.sites)} sites.'

