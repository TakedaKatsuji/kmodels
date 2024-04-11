from torch.utils.data import Dataset
from typing import Any
import cv2
from VGG.utils import DataCsvParser

class VGGDataset(Dataset):
    def __init__(self, csv_path: str, mode: str, transform=None) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.mode = mode
        self.transform = transform
        parser = DataCsvParser(csv_path=csv_path)
        self.data_path = parser.data_path(self.mode)
        self.data_label = parser.data_label(self.mode)
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index) -> Any:
        image = cv2.imread(self.data_path[index], 0)
        if self.transform:
            image = self.transform(image=image)["image"]
        label = self.data_label[index]
    
        return image, label
