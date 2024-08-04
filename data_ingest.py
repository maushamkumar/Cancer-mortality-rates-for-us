import pandas as pd


class IngestData:
    def __init__(self) -> None:
        self.data_path = None

    def get_data(self, data_path: str) -> pd.DataFrame:
        """
        > This function takes a path to a csv file and returns a pandas dataframe.

        :param data_path: The path to the data file
        :type data_path: str
        :return: A dataframe
        """
        self.data_path = data_path
        df = pd.read_csv(self.data_path)
        return df
