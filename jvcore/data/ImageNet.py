from jittor.dataset import ImageFolder
import os


class ImageNet(ImageFolder):
    def __init__(self, root, transform=None, train=False):
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        super().__init__(root, transform)
