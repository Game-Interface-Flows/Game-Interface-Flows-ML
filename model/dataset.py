import albumentations as A
import pytorch_lightning as pl
from albumentation_transforms import AlbumentationTransforms
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split
from torchvision import datasets


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int,
        train_test_ratio: float = 0.8,
        train_val_ratio: float = 0.8,
    ):

        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_test_ratio = train_test_ratio
        self.train_val_ratio = train_val_ratio

        self.transform = AlbumentationTransforms(
            A.Compose(
                [
                    A.Resize(width=256, height=256),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True),
                    A.ToGray(always_apply=True),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Normalize(mean=[0.485], std=[0.229]),
                    ToTensorV2(),
                ]
            )
        )

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            dataset = datasets.ImageFolder(self.data_path, transform=self.transform)
            # split dataset to train and test
            train_size = int(self.train_test_ratio * len(dataset))
            test_size = len(dataset) - train_size 
            # split train to train and val
            val_size = int((1 - self.train_val_ratio) * train_size)
            train_size = train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )

    def _get_data_loader(self, dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self._get_data_loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._get_data_loader(self.val_dataset)

    def test_dataloader(self):
        return self._get_data_loader(self.test_dataset)
