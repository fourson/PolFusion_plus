from torchvision import transforms
from torch.utils.data import DataLoader

from .dataset import dataset
from base.base_data_loader import BaseDataLoader


class WithGroundTruthDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
        ])
        self.dataset = dataset.WithGroundTruthDataset(data_dir, transform=transform)

        super(WithGroundTruthDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split,
                                                        num_workers)


class WithoutGroundTruthDataLoader(DataLoader):
    def __init__(self, data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
        ])
        self.dataset = dataset.WithoutGroundTruthDataset(data_dir, transform=transform)

        super(WithoutGroundTruthDataLoader, self).__init__(self.dataset)
