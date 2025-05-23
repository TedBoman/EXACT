import pandas as pd

class read_csv:
    def __init__(self, file_path):
        self.file_path = file_path

    def filetype_csv(self):
        """
        Processes a CSV file, injects anomalies, and inserts the data into the database.
        Ensures consistent anomaly injection across chunks.
        """

        full_df = pd.read_csv(self.file_path)

        return full_df
    
    def get_columns(self):
        # Read only the header row by specifying nrows=0
        df_header = pd.read_csv(self.file_path, nrows=0)

        # Get the column names as a list
        column_names = df_header.columns.tolist()
        return column_names