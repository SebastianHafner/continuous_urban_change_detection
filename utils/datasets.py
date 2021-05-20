import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

import json

from pathlib import Path
from abc import abstractmethod
from utils import geofiles, dataset_helpers, prediction_helpers, label_helpers, augmentations


class TrainingDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, run_type: str):
        super().__init__()
        self.cfg = cfg
        self.run_type = run_type

        self.root_path = Path(cfg.DATASETS.ROOT_PATH)
        self.dataset_name = cfg.DATASETS.NAME
        self.config_name = cfg.DATALOADER.CONFIG_NAME

        self.input = cfg.DATALOADER.INPUT
        self.output = cfg.DATALOADER.OUTPUT

        self.aoi_ids = cfg.DATASETS.AOI_IDS.TRAINING if run_type == 'training' else cfg.DATASETS.AOI_IDS.VALIDATION
        self.samples = []
        for aoi_id in self.aoi_ids:
            m, n = dataset_helpers.get_yx_size(self.dataset_name, aoi_id)
            for i in range(m):
                for j in range(n):
                    self.samples.append((aoi_id, i, j))
        self.length = len(self.samples)

        self.transform = augmentations.compose_transformations(cfg)

    def __getitem__(self, index: int) -> dict:

        aoi_id, i, j = self.samples[index]

        x = self._load_timeseries_data(aoi_id, i, j)
        y = self._load_label(aoi_id, i, j)

        x, y = self.transform((x, y))

        sample = {
            'x': x,
            'y': y,
            'aoi_id': aoi_id,
            'i': i,
            'j': j
        }

        return sample

    def _load_timeseries_data(self, aoi_id: str, i: int, j: int) -> np.ndarray:
        if self.input == 'prob':
            prob_cube = prediction_helpers.load_prediction_timeseries(self.dataset_name, aoi_id,
                                                                      dataset_helpers.include_masked())
            timeseries = prob_cube[i, j, ]
            timeseries = timeseries[:, np.newaxis]
        else:
            timeseries = None
        return timeseries

    def _load_label(self, aoi_id: str, i: int, j: int) -> np.ndarray:
        if self.output == 'change':
            label = label_helpers.generate_change_label(self.dataset_name, aoi_id, dataset_helpers.include_masked())
            label = np.array([label[i, j]])
            label = label[:, np.newaxis]
        else:
            label = None
        return label

    def __len__(self) -> int:
        return self.length

    def __str__(self) -> str:
        return 'not implemented'


class InferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, run_type: str):
        super().__init__()
        self.cfg = cfg
        self.run_type = run_type

        self.root_path = Path(cfg.DATASETS.ROOT_PATH)
        self.dataset_name = cfg.DATASETS.NAME
        self.config_name = cfg.DATALOADER.CONFIG_NAME

        self.input = cfg.DATALOADER.INPUT
        self.output = cfg.DATALOADER.OUTPUT

        self.aoi_ids = cfg.DATASETS.AOI_IDS.TRAINING if run_type == 'training' else cfg.DATASETS.AOI_IDS.VALIDATION
        self.length = len(self.aoi_ids)

        self.transform = augmentations.Numpy2Torch()

    def __getitem__(self, index: int) -> dict:

        aoi_id = self.aoi_ids[index]

        x = self._load_timeseries_data(aoi_id)
        y = self._load_label(aoi_id)

        x = TF.to_tensor(x)
        y = TF.to_tensor(y)

        sample = {
            'x': x,
            'y': y,
        }

        return sample

    def _load_timeseries_data(self, aoi_id: str) -> np.ndarray:
        if self.input == 'prob':
            prob_cube = prediction_helpers.load_prediction_timeseries(self.dataset_name, aoi_id,
                                                                      dataset_helpers.include_masked())
            timeseries = prob_cube
            timeseries = timeseries[:, np.newaxis]
        else:
            timeseries = None
        return timeseries

    def _load_label(self, aoi_id: str) -> np.ndarray:
        if self.output == 'change':
            label = label_helpers.generate_change_label(self.dataset_name, aoi_id, dataset_helpers.include_masked())
        else:
            label = None
        return label

    def __len__(self) -> int:
        return self.length

    def __str__(self) -> str:
        return 'not implemented'