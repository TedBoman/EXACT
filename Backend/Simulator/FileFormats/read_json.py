import pandas as pd

class read_json:
    def __init__(self, file_path):
        self.file_path = file_path

    def filetype_json(self):
        """
        Processes a json file, injects anomalies, and inserts the data into the database.
        Ensures consistent anomaly injection across chunks.
        """

        full_df = pd.read_json(self.file_path, orient='records')

        return full_df
    
    def get_columns(self):
        # Read only the header row by specifying nrows=0
        df_header = pd.read_json(self.file_path, orient='records')

        # Get the column names as a list
        column_names = df_header.columns.tolist()
        return column_names