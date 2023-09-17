from jittor.dataset import CIFAR10, CIFAR100, MNIST
from .ImageNet import ImageNet


def create_dataset(
        name,
        root,
        split='validation',
        transform=None,
        target_transform=None,
        download=False,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        **kwargs
):
    if split == 'validation':
        train = False
    else:
        train = True
    if name == 'CIFAR10':
        ds = CIFAR10(root=root, train=train, transform=transform, 
                     target_transform=target_transform,download=download, 
                     batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
        return ds
    elif name == 'CIFAR100':
        ds = CIFAR100(root=root, train=train, transform=transform, 
                     target_transform=target_transform,download=download, 
                     batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
        return ds
    elif name == 'MNIST':
        ds = MNIST(data_root=root, train=train, download=download, 
                   batch_size = batch_size, shuffle = shuffle, transform=transform, 
                   target_transform=target_transform, num_workers=num_workers, **kwargs)
        return ds
    elif name == 'ImageNet':
        ds = ImageNet(root=root, train=train, **kwargs).set_attrs(batch_size=batch_size, 
                                                                shuffle=shuffle, transform=transform, 
                                                                num_workers=num_workers)
        return ds
    else:
        print('Please implement the custom dataset')
        return None
    