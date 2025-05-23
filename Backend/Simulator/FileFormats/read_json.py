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