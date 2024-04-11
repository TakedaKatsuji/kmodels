import polars as pl

class DataCsvParser():
    def __init__(self, csv_path: str):
        import os
        import polars as pl
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pl.read_csv(csv_path)

    def data_path(self, mode):
        """
        特定のmodeに対応するpath列の値をリストとして返すプロパティ
        """
        try:
            return self.df.filter(pl.col("mode") == mode).get_column("path").to_list()
        except AttributeError:
            return []
    def data_label(self, mode):
        """
        特定のmodeに対応するラベル列をリストとして返すプロパティ
        """
        try:
            return self.df.filter(pl.col("mode")==mode).get_column("target").to_list()
        except AttributeError:
            return []
        
