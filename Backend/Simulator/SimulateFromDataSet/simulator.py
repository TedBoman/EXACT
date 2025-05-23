# simulator.py

import sys
import time as t
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

#from Simulator.DBAPI.db_interface import DBInterface as db
from timescaledb_api import TimescaleDBAPI as db
from Simulator.AnomalyInjector.anomalyinjector import TimeSeriesAnomalyInjector
import Simulator.DBAPI.utils as ut
from Simulator.DBAPI.debug_utils import DebugLogger as dl
from Simulator.FileFormats.read_csv import read_csv
from Simulator.FileFormats.read_json import read_json

class Simulator:
    """
    Simulates streaming data from a file to a database, with optional anomaly injection.

    Attributes:
        file_path (str): Path to the data file.
        file_extention (str): Extension of the data file.
        x_speedup (int, optional): Speedup factor for the simulation (default: 1).
        chunksize (int, optional): Chunk size for reading the file (default: 1).
        start_time (pd.Timestamp): Start time for the simulation.
    """

    def __init__(self, file_path, file_extention, start_time, x_speedup=1, chunksize=1):
        """
        Initializes Simulator with file path, extension, start time, speedup, and chunk size.
        """
        self.file_path = file_path
        self.file_extention = file_extention
        self.x_speedup = x_speedup
        self.chunksize = chunksize
        self.start_time = start_time

    def init_db(self, conn_params) -> db:
        """
        Returns an instance of the database interface API.

        Args:
            conn_params: A dictionary containing the parameters needed
                         to connect to the timeseries database.
        """
        db_instance = db(conn_params)
        return db_instance

    def create_table(self, conn_params, tb_name, columns):
        """
        Creates a table in the timeseries database. If the table already exists, 
        it creates a new table with a numbered suffix.

        Args: 
            conn_params: The parameters needed to connect to the database.
            tb_name: The base name of the table.
            columns: The columns for the table.

        Returns:
            str: The actual name of the table created (might include a suffix).
        """
        db_instance = self.init_db(conn_params)
        if not db_instance:
            return None

        i = 1
        new_table_name = tb_name
        while True:
            try:
                db_instance.create_table(new_table_name, columns)
                return new_table_name
            except psycopg2.errors.DuplicateTable:  # Catch the specific exception
                i += 1
                new_table_name = f"{tb_name}_{i}"
            except (psycopg2.errors.OperationalError, 
                    psycopg2.errors.ProgrammingError) as e:
                # Handle or log other database-related errors
                dl.print_exception(f"Database error creating table: {e}")
                raise  # Or re-raise if you want to stop execution
            except Exception as e:  # Catch other unexpected errors
                dl.print_exception(f"Unexpected error creating table: {e}")
                raise

    def process_row(self, conn_params, table_name, row, anomaly_settings=None):
        """
        Processes a single row of data, with optional anomaly injection.

        Args:
            conn_params: Database connection parameters
            table_name (str): Name of the table to insert data into
            row (pd.Series): A single row of data to be inserted
            anomaly_settings (list, optional): List of anomaly settings to apply
        """
        # Create a DataFrame from the single row
        df = pd.DataFrame([row])

        # Create a new column to track anomalies
        df['injected_anomaly'] = False
        df['is_anomaly'] = False
        
        dl.debug_print(anomaly_settings)

        injector = TimeSeriesAnomalyInjector()

        # Inject anomalies if settings are provided
        if anomaly_settings:
            for setting in anomaly_settings:
                dl.debug_print(setting)
                if setting.timestamp:
                    # Check if this row falls within the anomaly time range
                    row_timestamp = row['timestamp']

                    anomaly_start = setting.timestamp
                    anomaly_end = anomaly_start + pd.Timedelta(seconds=ut.parse_duration(setting.duration).total_seconds())
                    
                    dl.debug_print(anomaly_start)
                    dl.debug_print(anomaly_end)
                    dl.debug_print(row_timestamp)
                    dl.debug_print(df)

                    # Check if row timestamp is within anomaly time range
                    if anomaly_start <= row_timestamp <= anomaly_end:
                        dl.debug_print(f"Injecting anomaly on {row_timestamp}")
                        df = injector.inject_anomaly(df, setting)
                        dl.debug_print(df)

        db_instance = self.init_db(conn_params)
        db_instance.insert_data(table_name, df)
        dl.debug_print("Inserted row.")

    def start_simulation(self, conn_params, anomaly_settings=None, table_name=None, timestamp_col_name=None, label_col_name=None):
        """
        Reads the data file, preprocesses anomaly settings, and inserts data
        into the database row by row, with optional anomaly injection.

        Args:
            conn_params: Database connection parameters
            anomaly_settings (list, optional): List of anomaly settings to apply
        """
        dl.debug_print("Stream job has been started!")
        
        full_df = self.read_file()
        if full_df is None or full_df.empty:
            dl.debug_print(f"Fileformat {self.file_extention} not supported!")
            dl.debug_print("Canceling job")
            return
        
        #full_df = full_df.rename(columns={timestamp_col_name: 'timestamp'}) # Uncomment when passing timestamp column
        full_df.columns.values[0] = "timestamp"

        label_index = len(full_df.columns) - 1
        if label_index != None:
            full_df.columns.values[label_index] = "label"

        columns = list(full_df.columns.values)
                
        table_name = self.create_table(conn_params, Path(self.file_path).stem if table_name is None else table_name, columns)

        if table_name is None:
            dl.debug_print("Table could not be created!")
        else:
            dl.debug_print(f"Table {table_name} created!")

        # Preprocess anomaly settings to convert timestamps to absolute times
        if anomaly_settings:
            for setting in anomaly_settings:
                # Convert timestamp to absolute time if it's not already
                if not isinstance(setting.timestamp, pd.Timestamp):
                    setting.timestamp = self.start_time + pd.to_timedelta(setting.timestamp, unit='s').astype(np.int64) / 1e9
                
                if setting.columns:
                    setting.data_range = []
                    setting.mean = []
                    for col in setting.columns:
                        # Calculate and store the data range of this column
                        data_range = full_df[col].max() - full_df[col].min()
                        setting.data_range.append(data_range)

                        # Calculate and store the mean of this column
                        mean = full_df[col].mean()
                        setting.mean.append(mean)

        time_between_input = full_df.iloc[:, 0].diff().mean()
        dl.debug_print(f"Speedup: {self.x_speedup}")
        dl.debug_print(f"Time between inputs: {time_between_input} seconds")

        # Convert the first column (assume it's timestamps)
        try:
            # Try converting the first column to datetime objects directly
            full_df[full_df.columns[0]] = pd.to_datetime(full_df[full_df.columns[0]], unit='s')
        except ValueError:
            # If direct conversion fails, try converting to numeric first
            try:
                full_df[full_df.columns[0]] = pd.to_numeric(full_df[full_df.columns[0]])
                full_df[full_df.columns[0]] = pd.to_datetime(full_df[full_df.columns[0]], unit='s')  # Assuming seconds if numeric
            except ValueError:
                dl.print_exception("Error: Could not convert the first column to datetime. Please ensure it's in a valid format.")
                return

        # Calculate time difference in seconds
        time_between_input = full_df.iloc[:, 0].diff().dt.total_seconds().mean()

        # Drop rows with invalid timestamps
        full_df = full_df.dropna(subset=[full_df.columns[0]])

        dl.debug_print(f"Simulation speed between inputs: {time_between_input / self.x_speedup}")
        dl.debug_print("Starting to insert!")

        for index, row in full_df.iterrows():
            dl.debug_print(f"Inserting row {index + 1}")
            # Process the row with potential anomaly injection
            self.process_row(conn_params, table_name, row, anomaly_settings)

            # Sleep between rows, adjusted by speedup
            t.sleep(time_between_input / self.x_speedup)

        dl.debug_print("Inserting done!")
        
    def read_file(self):
        """
        Reads the data file based on its extension.

        Returns:
            pd.DataFrame: DataFrame containing the data from the file, or None 
                         if the file format is not supported.
        """
        try:
            match self.file_extention:
                case '.csv':
                    # File is a CSV file. Return a dataframe containing it.
                    csv = read_csv(self.file_path)
                    full_df = csv.filetype_csv()
                    return full_df
                case '.json':
                    json = read_json(self.file_path)
                    full_df = json.filetype_json()
                    return full_df
                # Add more fileformats here
                case _:
                    # Fileformat not supported
                    return None
        except Exception as e:
            print(f"Error: {e}")