from torch.utils.data import Dataset
from typing import Any
import cv2
from utils import DataCsvParser

class VGGDataset(Dataset):
    def __init__(self, csv_path: str, mode: str, transform=None) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.mode = mode
        self.transform = transform
        self.data_path = DataCsvParser(csv_path=csv_path).data_path(self.mode)
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index) -> Any:
        image = cv2.imread(self.data_path[index], 0)
        if self.transform is not None:
            image = self.transform(image)
        return image
