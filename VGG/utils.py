import polars as pl
import cv2


class DataCsvParser:
    def __init__(self, csv_path: str):
        import os
        import polars as pl

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pl.read_csv(csv_path)

    def get_data_list(self, mode):
        df = self.df.filter(pl.col("mode") == mode)
        image_path_list = df["path"].to_list()
        label_list = df["target"].to_list()
        image_list = [cv2.imread(path) for path in image_path_list]

        data_list = []
        for image, label in zip(image_list, label_list):
            data_list.append([image, label])

        return data_list
