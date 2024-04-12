from torch.utils.data import Dataset
from typing import Any
import cv2
from VGG.utils import DataCsvParser
import polars as pl


class VGGDataset(Dataset):
    def __init__(self, csv_path: str, mode: str, transform=None) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.mode = mode
        self.transform = transform

        parser = DataCsvParser(csv_path=csv_path)
        self.data = parser.get_data_list(mode=mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        image, label = self.data[index]
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
