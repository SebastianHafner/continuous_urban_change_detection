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
        img, label = args
        img_tensor = TF.to_tensor(img)
        label_tensor = TF.to_tensor(label)
        return img_tensor, label_tensor


class RandomNoise(object):
    def __call__(self, args):
        img, label = args

        return img, label
