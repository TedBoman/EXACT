# batchimport.py
import sys
import traceback
import psycopg2
import time as t
import multiprocessing as mp
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import pytz

#from Simulator.DBAPI.db_interface import DBInterface as db
from timescaledb_api import TimescaleDBAPI as db
from Simulator.AnomalyInjector.anomalyinjector import TimeSeriesAnomalyInjector
import Simulator.DBAPI.utils as ut
from Simulator.DBAPI.debug_utils import DebugLogger as dl
from Simulator.FileFormats.read_csv import read_csv
from Simulator.FileFormats.read_json import read_json

class BatchImporter:
    """
    Imports data from a file into a database in batches, with optional anomaly injection.

    Attributes:
        file_path (str): Path to the data file.
        file_extention (str): Extension of the data file.
        start_time (pd.Timestamp): Start time for the data.
        chunksize (int, optional): Size of each batch (default: 100).
    """

    def __init__(self, file_path, file_extention, start_time, chunksize=100):
        """
        Initializes BatchImporter with file path, extension, start time, and chunk size.
        """
        self.file_path = file_path
        self.chunksize = chunksize
        self.start_time = start_time
        self.file_extention = file_extention

    def init_db(self, conn_params) -> db:
        """
        Returns an instance of the database interface API.

        Args:
            conn_params: A dictionary containing the parameters needed
                         to connect to the timeseries database.
        """
        retry = 0

        while retry < 5:
            db_instance = db(conn_params)
            if db_instance:
                return db_instance
            else:
                time = 3
                while time > 0:
                    dl.debug_print("Retrying in: {time}s")
                    t.sleep(1)
        return None

    def create_table(self, conn_params, tb_name, columns):
        """
        Ensures a table exists in the timeseries database using the DatabaseAPI.

        Args:
            tb_name: The desired base name of the table.
            columns: The columns for the table.

        Returns:
            str: The actual name of the table if newly created.
            int: 1 if the table already exists.
            None: If an error occurred during the process or DB connection failed.
        """
        db_instance = self.init_db(conn_params)
        if not db_instance:
            return None

        result = db_instance.create_table(tb_name, list(columns))

        if result == None:
            # Table already exists, let the calling function know
            return None
        elif isinstance(result, str):
            # Table was created successfully, return the name
            return result
        else: # result is None (error)
            print(f"Error reported by API during creation of table '{tb_name}'.")
            return None
        
    def process_chunk(self, conn_params, table_name, chunk_df):
        # This function is called by multiprocessing, ensure it's self-contained or pickles correctly.
        # dl.debug_print(f"Process {mp.current_process().pid}: Processing chunk for table {table_name}. Chunk shape: {chunk_df.shape}")
        
        if not isinstance(chunk_df, pd.DataFrame) or chunk_df.empty:
            dl.debug_print(f"Process {mp.current_process().pid}: Chunk for table {table_name} is empty or not a DataFrame. Nothing to insert.")
            return

        # Critical check: Ensure timestamp column is present and in correct UTC format
        if 'timestamp' not in chunk_df.columns:
            dl.debug_print(f"CRITICAL ERROR in process_chunk: 'timestamp' column missing in chunk for table {table_name}. Skipping insertion.")
            return
        if not pd.api.types.is_datetime64_any_dtype(chunk_df['timestamp']):
            dl.debug_print(f"CRITICAL ERROR in process_chunk: 'timestamp' column in chunk for table {table_name} is not datetime64. Dtype: {chunk_df['timestamp'].dtype}. Skipping insertion.")
            return
        if chunk_df['timestamp'].dt.tz is None or str(chunk_df['timestamp'].dt.tz).upper() != 'UTC':
            dl.debug_print(f"CRITICAL WARNING in process_chunk: Timestamp for {table_name} not UTC. TZ: {chunk_df['timestamp'].dt.tz}. Attempting conversion.")
            try:
                if chunk_df['timestamp'].dt.tz is None:
                    chunk_df.loc[:, 'timestamp'] = chunk_df['timestamp'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                else:
                    chunk_df.loc[:, 'timestamp'] = chunk_df['timestamp'].dt.tz_convert('UTC')
                chunk_df.dropna(subset=['timestamp'], inplace=True) # Drop if localization/conversion failed
                if chunk_df.empty:
                    dl.debug_print(f"Process {mp.current_process().pid}: Chunk became empty after UTC conversion for {table_name}.")
                    return
            except Exception as e_ts_conv:
                dl.print_exception(f"Error converting chunk timestamp to UTC for {table_name}: {e_ts_conv}. Skipping insertion.")
                return
        
        db_instance = self.init_db(conn_params) # init_db is part of BatchImporter, might need to be static or passed
        if not db_instance:
            dl.debug_print(f"Process {mp.current_process().pid}: DB connection failed in process_chunk for table {table_name}. Chunk not inserted.")
            return 
            
        try:
            if not chunk_df.empty:
                # dl.debug_print(f"Process {mp.current_process().pid}: Inserting {len(chunk_df)} rows into {table_name}.")
                db_instance.insert_data(table_name, chunk_df)
            # else:
                # dl.debug_print(f"Process {mp.current_process().pid}: Chunk for table {table_name} is empty. Nothing to insert.")
        except Exception as e_insert:
            dl.print_exception(f"Process {mp.current_process().pid}: Error inserting data into {table_name}: {e_insert}")

    def inject_anomalies_into_chunk(self, chunk_df_input: pd.DataFrame, 
                                    anomaly_settings_abs_utc: list, 
                                    chunk_index_for_logging="N/A") -> pd.DataFrame:
        
        dl.debug_print(f"BatchImporter.inject_anomalies_into_chunk for chunk {chunk_index_for_logging}. Input chunk shape: {chunk_df_input.shape}")
        sys.stdout.flush()

        if not isinstance(chunk_df_input, pd.DataFrame) or chunk_df_input.empty:
            dl.debug_print(f"  Chunk {chunk_index_for_logging} is empty, skipping anomaly injection.")
            sys.stdout.flush()
            return chunk_df_input
        if not anomaly_settings_abs_utc:
            return chunk_df_input
        
        injector = TimeSeriesAnomalyInjector() 
        current_modified_chunk = chunk_df_input 

        for i, setting in enumerate(anomaly_settings_abs_utc):
            injected_sum_before_this_setting = current_modified_chunk['injected_anomaly'].sum() if 'injected_anomaly' in current_modified_chunk else 0
            try:
                chunk_after_this_setting = injector.inject_anomaly(current_modified_chunk, setting) 
                
                injected_sum_after_this_setting = chunk_after_this_setting['injected_anomaly'].astype(bool).sum() if 'injected_anomaly' in chunk_after_this_setting else 0
                injected_sum_before_this_setting_bool = current_modified_chunk['injected_anomaly'].astype(bool).sum() if 'injected_anomaly' in current_modified_chunk else 0
                num_newly_flagged_by_this_setting = injected_sum_after_this_setting - injected_sum_before_this_setting_bool


                if num_newly_flagged_by_this_setting > 0:
                    dl.debug_print(f"    Processed setting {i} ('{setting.anomaly_type}'): {num_newly_flagged_by_this_setting} new 'injected_anomaly' flags set by injector in chunk {chunk_index_for_logging}.")
                elif injected_sum_after_this_setting > 0 and injected_sum_before_this_setting_bool == 0 :
                     dl.debug_print(f"    Processed setting {i} ('{setting.anomaly_type}'): 'injected_anomaly' flags set from zero by injector in chunk {chunk_index_for_logging}.")
                sys.stdout.flush()
                current_modified_chunk = chunk_after_this_setting 
            except Exception as e_injector_call:
                dl.print_exception(f"  Error calling TimeSeriesAnomalyInjector.inject_anomaly for setting {i} on chunk {chunk_index_for_logging}: {e_injector_call}")
                sys.stdout.flush()
        
        return current_modified_chunk
    
    
    def start_simulation(self, conn_params, anomaly_settings=None, table_name=None, timestamp_col_name=None, label_col_name=None):
        """
        Starts the batch data import process.

        Reads the data file, preprocesses anomaly settings, and inserts data
        into the database in chunks, with optional anomaly injection.

        Args:
            conn_params: Database connection parameters.
            anomaly_settings (list, optional): List of anomaly settings to apply.
        """
        num_processes = mp.cpu_count()
        pool = mp.Pool(processes=num_processes)

        dl.debug_print(self.file_path)
        dl.debug_print(self.chunksize)
        dl.debug_print(self.start_time)
        dl.debug_print("Starting to insert!")
        
        full_df = self.read_file()
        if full_df is None or full_df.empty:
            dl.print_exception(f"Fileformat {self.file_extention} not supported!")
            dl.print_exception("Canceling job")
            return
        
        # --- Drop Unnamed Columns ---
        # Select columns that do not start with 'Unnamed:'
        full_df = full_df.loc[:, ~full_df.columns.str.startswith('Unnamed:')]
        dl.debug_print("Dropped unnamed columns.")
        # --- End Drop Unnamed Columns ---
        
        rename_map = {}
        if label_col_name != 'label':
            rename_map[label_col_name] = 'label'
        if timestamp_col_name != 'timestamp':
            rename_map[timestamp_col_name] = 'timestamp'

        # --- Perform Renaming Operation ---
        if rename_map: # Only rename if there's anything to rename
            full_df = full_df.rename(columns=rename_map)
            dl.debug_print("Label column renaming applied.")
        
        # --- DataFrame Timestamp Conversion ---
        if 'timestamp' in full_df.columns:
            timestamp_col = full_df['timestamp']
            if pd.api.types.is_numeric_dtype(timestamp_col):
                dl.debug_print("Numeric 'timestamp' column in DataFrame. Interpreting as seconds since Unix epoch.")
                # Using pd.to_numeric to handle potential strings that are numbers, then to_datetime
                full_df.loc[:, 'timestamp'] = pd.to_datetime(pd.to_numeric(timestamp_col, errors='coerce'), unit='s', utc=True, errors='coerce')
            elif not pd.api.types.is_datetime64_any_dtype(timestamp_col): # It's object/string etc.
                dl.debug_print("Non-numeric, non-datetime 'timestamp' column in DataFrame. Attempting to parse as datetime strings.")
                full_df.loc[:, 'timestamp'] = pd.to_datetime(timestamp_col, utc=True, errors='coerce')
            else: # Already datetime64; ensure UTC
                dl.debug_print("'timestamp' column is already datetime. Ensuring it is UTC.")
                if timestamp_col.dt.tz is None:
                    full_df.loc[:, 'timestamp'] = timestamp_col.dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                elif str(timestamp_col.dt.tz).upper() != 'UTC':
                    full_df.loc[:, 'timestamp'] = timestamp_col.dt.tz_convert('UTC')
            
            # Handle NaTs from conversion
            initial_rows = len(full_df)
            full_df.dropna(subset=['timestamp'], inplace=True)
            if len(full_df) < initial_rows:
                dl.debug_print(f"Dropped {initial_rows - len(full_df)} rows due to NaT timestamps after conversion.")
            if full_df.empty:
                dl.debug_print("DataFrame became empty after dropping NaT timestamps. Canceling job.")
                pool.close()
                pool.join()
                return 0 # Indicate failure
        else:
            dl.debug_print("CRITICAL: 'timestamp' column not found in DataFrame. Cannot proceed.")
            pool.close()
            pool.join()
            return 0 # Indicate failure

        dl.debug_print("Sample of DataFrame after all timestamp processing (should be datetime64[ns, UTC]):")
        dl.debug_print(full_df.head())
            
        # --- Label Column Conversion to 0/1 ---
        if 'label' in full_df.columns:
            dl.debug_print(f"Processing 'label' column. Initial unique values (up to 10): {full_df['label'].dropna().unique()[:10]}")

            def convert_label_value(val):
                if isinstance(val, str):
                    val_lower = val.lower()
                    if val_lower == 'true':
                        return 1
                    if val_lower == 'false':
                        return 0
                elif isinstance(val, bool):
                    return 1 if val else 0
                
                # Check for numeric 1 or 0 (handles int and float)
                # Using np.isclose for float comparison is safer if precision issues were a concern,
                # but direct equality works for exact 0.0 and 1.0.
                if val == 1 or val == 1.0:
                    return 1
                if val == 0 or val == 0.0:
                    return 0
                
                # Default for anything else (other numbers, unhandled strings, None, NaN)
                return 0

            full_df['label'] = full_df['label'].apply(convert_label_value).astype(int)
            
            dl.debug_print(f"Converted 'label' column to 0/1 integers. Unique values after conversion: {full_df['label'].unique()}")
        else:
            dl.debug_print("'label' column not found or specified. Skipping label conversion.")

        # Sort by timestamp to ensure iloc[0] is the earliest if not already sorted.
        full_df.sort_values(by='timestamp', inplace=True)
        dl.debug_print("DataFrame sorted by 'timestamp' column.")

        # ***** SET self.start_time FROM THE FIRST VALID TIMESTAMP *****
        self.start_time = full_df['timestamp'].iloc[0]
        dl.debug_print(f"Data-derived self.start_time set to: {self.start_time}")

        # Preprocess anomaly settings to convert timestamps to absolute UTC times
        if anomaly_settings:
            dl.debug_print(f"Preprocessing {len(anomaly_settings)} anomaly settings for absolute UTC timestamps...")
            for i, setting in enumerate(anomaly_settings):
                original_setting_ts = setting.timestamp # For logging
                
                dl.debug_print(f"Setting.timestamp: {setting.timestamp}")
                dl.debug_print(f"start_time: {self.start_time}")

                # Step 1: Convert to absolute pd.Timestamp if it's a relative offset
                if not isinstance(setting.timestamp, pd.Timestamp):
                    try:
                        # Assuming setting.timestamp is a numeric offset (int or string convertible to float)
                        # representing seconds from self.start_time
                        time_offset_seconds = float(str(setting.timestamp)) # Ensure it's floatable
                        setting.timestamp = self.start_time + pd.to_timedelta(time_offset_seconds, unit='s')
                        # dl.debug_print(f"  Anomaly setting {i}: Relative '{original_setting_ts}' + start_time '{self.start_time}' -> Absolute '{setting.timestamp}'")
                    except ValueError as ve:
                        dl.debug_print(f"  WARNING: Anomaly setting {i}: Could not convert timestamp offset '{original_setting_ts}' to timedelta. Error: {ve}. Skipping UTC conversion for this timestamp.")
                        continue # Skip to next setting if offset conversion fails
                    except TypeError as te: # E.g. if self.start_time is None
                        dl.debug_print(f"  WARNING: Anomaly setting {i}: TypeError during offset calculation for timestamp '{original_setting_ts}' (start_time: {self.start_time}). Error: {te}. Skipping.")
                        continue


                # Step 2: Ensure the (now absolute) setting.timestamp is UTC-aware
                if isinstance(setting.timestamp, pd.Timestamp):
                    if setting.timestamp.tzinfo is None:
                        # If self.start_time was naive, the resulting absolute timestamp is naive.
                        # Assume it should be interpreted as UTC.
                        try:
                            setting.timestamp = setting.timestamp.tz_localize('UTC')
                            # dl.debug_print(f"  Anomaly setting {i}: Localized naive timestamp to UTC: '{setting.timestamp}'")
                        except (pytz.exceptions.AmbiguousTimeError, pytz.exceptions.NonExistentTimeError) as tze:
                            # This can happen if the naive timestamp falls on a DST transition
                            dl.debug_print(f"  WARNING: Anomaly setting {i}: Failed to localize naive timestamp {setting.timestamp} to UTC due to DST ambiguity/non-existence: {tze}. Attempting 'infer' or safe option.")
                            try: # Try to handle DST transitions carefully
                                setting.timestamp = setting.timestamp.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                                if pd.isna(setting.timestamp):
                                    dl.debug_print(f"  ERROR: Anomaly setting {i}: Timestamp {original_setting_ts} resulted in NaT after tz_localize. Cannot use.")
                                    # Potentially mark this setting as invalid or remove it
                                    continue
                            except Exception as e_loc_fallback:
                                dl.debug_print(f"  ERROR: Anomaly setting {i}: Critical error localizing timestamp {original_setting_ts}: {e_loc_fallback}")
                                continue

                    elif str(setting.timestamp.tzinfo).upper() != 'UTC':
                        # If it's already tz-aware but not UTC, convert it to UTC.
                        try:
                            setting.timestamp = setting.timestamp.tz_convert('UTC')
                            # dl.debug_print(f"  Anomaly setting {i}: Converted timestamp from {original_setting_ts.tzinfo} to UTC: '{setting.timestamp}'")
                        except Exception as e_conv:
                            dl.debug_print(f"  WARNING: Anomaly setting {i}: Failed to convert timestamp {original_setting_ts} to UTC: {e_conv}")
                            continue
                    # else: It's already a pd.Timestamp and UTC-aware, no change needed.
                    #    dl.debug_print(f"  Anomaly setting {i}: Timestamp '{setting.timestamp}' is already UTC-aware.")

                else:
                    dl.debug_print(f"  WARNING: Anomaly setting {i}: Its 'timestamp' attribute is not a pd.Timestamp after processing (type: {type(setting.timestamp)}, value: {setting.timestamp}). Cannot ensure UTC.")
                    continue # Skip if not a valid timestamp object
        # Create a list to store results from async processes
        results = []
        
        columns = list(full_df.columns.values)
        
        table_name = self.create_table(conn_params, Path(self.file_path).stem if table_name is None else table_name, columns)
        if table_name == None:
            return 1

        # Drop rows with invalid timestamps
        full_df = full_df.dropna(subset=[full_df.columns[0]])

        dl.debug_print(full_df.head())  # Inspect the parsed DataFrame

        # Set the chunksize to the number of rows in the file / cpu cores available
        self.chunksize = len(full_df.index) / num_processes

        # Create a new column to track anomalies
        full_df['injected_anomaly'] = False
        full_df['is_anomaly'] = False
        
        current_chunksize = int(self.chunksize) if self.chunksize is not None and self.chunksize > 0 else 10000 
        if len(full_df) / current_chunksize > num_processes * 10 and len(full_df) > 100000 : 
            current_chunksize = int(np.ceil(len(full_df) / (num_processes * 5)))
        
        results = []
        chunk_list = [full_df[i:i + current_chunksize] for i in range(0, len(full_df), current_chunksize)]
        dl.debug_print(f"Split DataFrame into {len(chunk_list)} chunks.")
        sys.stdout.flush()

        # Process chunks with anomaly injection
        for idx, chunk_original_slice in enumerate(chunk_list):
            chunk_to_process = chunk_original_slice.copy()
            if anomaly_settings:
                # Inject anomalies across chunk boundaries
                chunk_to_process = self.inject_anomalies_into_chunk(chunk_to_process, anomaly_settings,chunk_index_for_logging=str(idx+1))
                if 'injected_anomaly' in chunk_to_process.columns:
                    if 'is_anomaly' not in chunk_to_process.columns: 
                        chunk_to_process.loc[:, 'is_anomaly'] = False
                    is_anomaly_bool = chunk_to_process['is_anomaly'].astype(bool)
                    injected_anomaly_bool = chunk_to_process['injected_anomaly'].astype(bool)
                    chunk_to_process.loc[:, 'is_anomaly'] = (is_anomaly_bool | injected_anomaly_bool) # Result of OR is boolean
                
                dl.debug_print(f"  Chunk {idx+1} PREPARED for worker. injected_anomaly sum: {chunk_to_process['injected_anomaly'].sum()}, is_anomaly sum: {chunk_to_process['is_anomaly'].sum()}")
                sys.stdout.flush()
                
            # Use apply_async and collect results
            result = pool.apply_async(
                self.process_chunk,
                args=(conn_params, table_name, chunk_to_process),
            )
            results.append(result)

        # Wait for all processes to complete
        pool.close()
        pool.join()

        # Check for any exceptions in the results
        for result in results:
            result.get()  # This will raise any exceptions that occurred in the process

        dl.debug_print("Inserting done!")
        return 1
        

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
