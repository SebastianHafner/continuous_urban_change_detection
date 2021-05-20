import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np


def compose_transformations(cfg):
    transformations = []

    if cfg.AUGMENTATION.NOISE:
        transformations.append(RandomNoise())

    transformations.append(Numpy2Torch())

    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, args):
        timeseries, label = args
        timeseries_tensor = TF.to_tensor(timeseries)
        label_tensor = TF.to_tensor(label)
        return timeseries_tensor[0, ], label_tensor[0, 0, ]


class RandomNoise(object):
    def __call__(self, args):
        img, label = args

        return img, label
