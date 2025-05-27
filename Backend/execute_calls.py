import json
import sys
import traceback
import warnings
import pandas as pd
import numpy as np
import os
import time
from socket import socket
from timescaledb_api import TimescaleDBAPI
from datetime import datetime, timezone
from typing import Any

# Third-Party
from pathlib import Path

import pandas as pd
from typing import List, Optional, Dict

# Custom
from Simulator.DBAPI.type_classes import Job
from Simulator.DBAPI.type_classes import AnomalySetting
from Simulator.SimulatorEngine import SimulatorEngine as se
from ML_models.get_model import get_model

# --- XAI ---
from ML_models.model_wrapper import ModelWrapperForXAI
from XAI_methods.xai_runner import XAIRunner

MODEL_DIRECTORY = "./ML_models"
INJECTION_METHOD_DIRECTORY = "./Simulator/AnomalyInjector/InjectionMethods"
XAI_METHOD_DIRECTORY = "/XAI_methods/methods"
DATASET_DIRECTORY = "./Datasets"
OUTPUT_DIR = "/data"

UNSUPERVISED_MODELS = [
    'lstm',
    'svm',
    'isolation_forest',
]

PRIMARY_KEY_COLUMN = 'id'

# --- Utility function to save results ---
def save_run_summary(summary_dict: Dict[str, Any], job_name: str, output_dir: str) -> None:
    """Appends a run summary dictionary to a JSON Lines file."""
    try:
        # Ensure output directory exists
        
        output_path = output_dir+'/'+job_name+'/'+'logfile'

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert complex objects for JSON serialization
        serializable_summary = {}
        for k, v in summary_dict.items():
            if isinstance(v, np.ndarray):
                serializable_summary[k] = v.tolist()
            elif isinstance(v, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): # Handle numpy integers
                 serializable_summary[k] = int(v)
            elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)): # Handle numpy floats
                 serializable_summary[k] = float(v)
            elif isinstance(v, pd.Timestamp): # Handle pandas timestamp
                serializable_summary[k] = v.tz_convert(None).isoformat() if v.tz is not None else v.isoformat() # Ensure timezone naive for ISO format consistency if needed, or keep tz
            elif isinstance(v, datetime):
                 serializable_summary[k] = v.isoformat()
            # Add other type conversions if needed (e.g., Path objects)
            elif isinstance(v, Path):
                serializable_summary[k] = str(v)
            else:
                serializable_summary[k] = v

        json_string = json.dumps(serializable_summary, default=str) # Use default=str as fallback
        with open(output_path, 'w') as f:
            f.write(json_string + '\n')
        print(f"Successfully saved run summary to {output_path}")
    except Exception as e:
        print(f"Error saving run summary to {output_path}: {e}")
        traceback.print_exc()

def get_anomaly_rows(
    data: pd.DataFrame,
    label_column: str = 'label', # Common name for the label column
    anomaly_value: Any = 1       # The value indicating an anomaly (often 1 or True)
) -> pd.DataFrame:
    """
    Filters a pandas DataFrame to return only the rows marked as anomalies
    based on a specific label column and value.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data and labels.
        label_column (str): The name of the column containing the anomaly labels.
                            Defaults to 'label'. Common alternatives might be
                            'is_anomaly', 'anomaly', 'target'.
        anomaly_value (Any): The value within the `label_column` that signifies
                             an anomaly. Defaults to 1. This could also be True,
                             'anomaly', etc., depending on your dataset.

    Returns:
        pd.DataFrame: A new DataFrame containing only the rows from the input `data`
                      where the value in the `label_column` equals `anomaly_value`.
                      Preserves the original index and columns. Returns an empty
                      DataFrame with the same columns if no anomalies are found
                      or if the input DataFrame is empty.

    Raises:
        TypeError: If 'data' is not a pandas DataFrame.
        ValueError: If 'label_column' is not found in the DataFrame's columns.
    """
    # 1. Input Validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")

    if label_column not in data.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in DataFrame columns. "
            f"Available columns: {data.columns.tolist()}"
        )

    if data.empty:
        print("Input DataFrame is empty. Returning an empty DataFrame.")
        # Return an empty frame structure matching the input
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)

    # 2. Filtering Logic
    try:
        # Create a boolean mask where the condition is true
        anomaly_mask = (data[label_column] == anomaly_value)

        # Use the mask to select rows. Use .copy() to avoid potential
        # SettingWithCopyWarning if the returned DataFrame is modified later.
        anomaly_rows = data.loc[anomaly_mask].copy()

        #print(f"Found {len(anomaly_rows)} rows where '{label_column}' == {anomaly_value}.")

    except Exception as e:
        # Catch potential errors during comparison or indexing
        print(f"An error occurred during filtering: {e}")
        # Return an empty DataFrame with original structure on error
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)

    # 3. Return Filtered DataFrame
    return anomaly_rows

def get_balanced_anomaly_sample(
    data: pd.DataFrame,
    total_rows: int, # Parameter for the desired total number of rows
    label_column: str = 'label',
    anomaly_value: Any = 1,
    random_state: Optional[int] = None # Still useful for internal consistency if needed, but sampling is not random
) -> pd.DataFrame:
    """
    Filters a pandas DataFrame to return a new DataFrame containing a sample
    of anomaly and non-anomaly rows, aiming for a balanced distribution,
    up to a specified total number of rows, while preserving temporal order
    by sampling from the end of each sequence.

    The function samples the most recent `N_anomaly` anomalies and the most
    recent `N_non_anomaly` non-anomalies, where N_anomaly + N_non_anomaly
    equals the minimum of:
    1. The requested `total_rows`.
    2. The total number of available rows in the input `data`.
    3. Twice the count of the less frequent class (to maintain balance if possible).

    The function prioritizes achieving the `total_rows` while keeping the
    anomaly/non-anomaly split as close to 50/50 as possible given the constraints.
    The output DataFrame will contain the sampled anomalies followed by the
    sampled non-anomalies, maintaining their original relative order within
    each group.

    Args:
        data (pd.DataFrame): 
                            The input DataFrame containing the data and labels.
                            Assumes data is already sorted by time/index.
        total_rows (int): 
                        The desired total number of rows in the output DataFrame.
                        Must be a non-negative integer.
        label_column (str): The name of the column containing the anomaly labels.
                            Defaults to 'label'.
        anomaly_value (Any): 
                            The value within the `label_column` that signifies
                            an anomaly. Defaults to 1.
        random_state (Optional[int]): 
                                    Seed for any potential internal random
                                    operations (though primary sampling is
                                    now temporal).

    Returns:
        pd.DataFrame: 
                A new DataFrame containing the sampled rows, preserving
                temporal order within each class and concatenating them.
                Preserves original columns. Returns an empty DataFrame
                with the same columns if the input is empty, total_rows
                is 0, or no instances of a required class are available.

    Raises:
        TypeError: If 'data' is not a pandas DataFrame or total_rows is not an int.
        ValueError: If 'label_column' is not found or total_rows is negative.
    """
    # 1. Input Validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")

    if not isinstance(total_rows, int):
        raise TypeError("Input 'total_rows' must be an integer.")

    if total_rows < 0:
        raise ValueError("Input 'total_rows' must be a non-negative integer.")

    if label_column not in data.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in DataFrame columns. "
            f"Available columns: {data.columns.tolist()}"
        )

    if data.empty or total_rows == 0:
        print("Input DataFrame is empty or total_rows is 0. Returning an empty DataFrame.")
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)

    # Note: random_state is less critical now as sampling is temporal,
    # but kept for potential future use or internal pandas operations.
    np.random.seed(random_state)

    # 2. Separate Anomalies and Non-Anomalies (preserving original order)
    try:
        anomaly_mask = (data[label_column] == anomaly_value)
        # Use .loc and keep original order
        anomaly_df = data.loc[anomaly_mask].copy()
        non_anomaly_df = data.loc[~anomaly_mask].copy() # Use inverse mask

        n_anomalies = len(anomaly_df)
        n_non_anomalies = len(non_anomaly_df)

        print(f"Found {n_anomalies} anomaly rows and {n_non_anomalies} non-anomaly rows.")

    except Exception as e:
        print(f"An error occurred during data separation: {e}")
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)

    # Check if any class is empty before proceeding
    if n_anomalies == 0 and n_non_anomalies == 0:
        print("Input DataFrame contains no rows. Returning empty DataFrame.")
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)
    elif n_anomalies == 0:
        print("No anomaly instances found. Returning up to total_rows from the end of non-anomalies.")
        # If no anomalies, return up to total_rows from the end of non-anomalies
        num_non_anomalies_to_sample = min(total_rows, n_non_anomalies)
        # Use .tail() to get the last rows
        sampled_df = non_anomaly_df.tail(n=num_non_anomalies_to_sample)
        # Reset index but DO NOT shuffle
        return sampled_df.reset_index(drop=True)
    elif n_non_anomalies == 0:
        print("No non-anomaly instances found. Returning up to total_rows from the end of anomalies.")
        # If no non-anomalies, return up to total_rows from the end of anomalies
        num_anomalies_to_sample = min(total_rows, n_anomalies)
        # Use .tail() to get the last rows
        sampled_df = anomaly_df.tail(n=num_anomalies_to_sample)
        # Reset index but DO NOT shuffle
        return sampled_df.reset_index(drop=True)


    # 3. Determine Actual Number of Rows to Return
    actual_total_to_return = min(total_rows, n_anomalies + n_non_anomalies)
    print(f"Aiming to return {actual_total_to_return} rows (min of requested {total_rows} and available {n_anomalies + n_non_anomalies}).")


    # 4. Determine Number of Samples from Each Class (Aiming for Balance)
    # Calculate target counts for a 50/50 split
    target_anomalies = actual_total_to_return // 2
    target_non_anomalies = actual_total_to_return - target_anomalies # Handles odd total

    # Calculate initial sample counts, limited by available data
    num_anomalies_to_sample = min(target_anomalies, n_anomalies)
    num_non_anomalies_to_sample = min(target_non_anomalies, n_non_anomalies)

    # Calculate how many more samples are needed to reach actual_total_to_return
    current_total = num_anomalies_to_sample + num_non_anomalies_to_sample
    remaining_needed = actual_total_to_return - current_total

    # Distribute the remaining needed samples to the class with more available capacity
    # This ensures we hit actual_total_to_return while staying within class limits
    if remaining_needed > 0:
        anomaly_capacity_left = n_anomalies - num_anomalies_to_sample
        non_anomaly_capacity_left = n_non_anomalies - num_non_anomalies_to_sample

        # Prioritize adding to the class that has more available instances remaining
        if anomaly_capacity_left > non_anomaly_capacity_left:
            add_anomalies = min(remaining_needed, anomaly_capacity_left)
            num_anomalies_to_sample += add_anomalies
            remaining_needed -= add_anomalies

        if remaining_needed > 0:
            add_non_anomalies = min(remaining_needed, non_anomaly_capacity_left)
            num_non_anomalies_to_sample += add_non_anomalies
            # remaining_needed should now be 0

    print(f"Sampling {num_anomalies_to_sample} anomalies and {num_non_anomalies_to_sample} non-anomalies from the end of each sequence.")


    # 5. Sample from the End of Each Group (Temporal Sampling)
    try:
        # Use .tail() to get the last 'n' rows, preserving their order
        sampled_anomalies = anomaly_df.tail(n=num_anomalies_to_sample)
        sampled_non_anomalies = non_anomaly_df.tail(n=num_non_anomalies_to_sample)

    except Exception as e:
        print(f"An unexpected error occurred during temporal sampling: {e}")
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)


    # 6. Combine (DO NOT SHUFFLE)
    try:
        # Concatenate the two sampled DataFrames. The order will be
        # sampled_anomalies followed by sampled_non_anomalies.
        balanced_sample_df = pd.concat([sampled_anomalies, sampled_non_anomalies])

        # Reset index to get a clean 0-based index for the new combined DataFrame,
        # but explicitly do not shuffle.
        balanced_sample_df = balanced_sample_df.reset_index(drop=True)

        print(f"Created temporal sample with {len(balanced_sample_df)} rows.")
        if not balanced_sample_df.empty:
            print("Value counts in sample:\n", balanced_sample_df[label_column].value_counts())

    except Exception as e:
        print(f"An error occurred during concatenation or index reset: {e}")
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)


    # 7. Return Sampled DataFrame
    return balanced_sample_df

def split_data(data):
    """Split the dataseries into 2 data series in a ratio of """

    total_rows = len(data)

    # Calculate split indices

    train_end = int(total_rows * 0.85) # 85% for training

    # Split the data
    training_data = data.iloc[:train_end]
    testing_data = data.iloc[train_end:] # Remaining 15% is testing

    return training_data, testing_data

def map_to_timestamp(time):
    return time.timestamp()

def map_to_time(time):
    return datetime.fromtimestamp(time, tz=timezone.utc)

def evaluate_classification(df: pd.DataFrame) -> dict:
    # Ensure the DataFrame has the necessary columns
    if "is_anomaly" not in df.columns:
        raise ValueError("DataFrame must contain an 'is_anomaly' column.")
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column.")

    # Convert boolean columns to integers for easier comparison (True=1, False=0)
    df["predicted"] = df["is_anomaly"].astype(int)
    df["actual"] = df["label"].astype(int)

    # Calculate evaluation metrics
    correct_anomalies = df[(df["predicted"] == 1) & (df["actual"] == 1)].shape[0]
    correct_non_anomalies = df[(df["predicted"] == 0) & (df["actual"] == 0)].shape[0]
    false_positives = df[(df["predicted"] == 1) & (df["actual"] == 0)].shape[0]
    false_negatives = df[(df["predicted"] == 0) & (df["actual"] == 1)].shape[0]

    total_predictions = len(df)
    accuracy = (correct_anomalies + correct_non_anomalies) / total_predictions if total_predictions > 0 else 0

    return {
        "correct_anomalies": correct_anomalies,
        "correct_non_anomalies": correct_non_anomalies,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_predictions": total_predictions,
        "accuracy": accuracy,
    }

# Starts processing of dataset in one batch
def run_batch(
    db_conn_params: Dict,
    model: str,                          # Model name string (e.g., 'lstm', 'svm')
    path: str,                           # Path to dataset file
    name: str,                           # Job name (e.g., "job_batch_myjob")
    inj_params: Optional[List[Dict[str, Any]]] = None, 
    debug: bool = False,
    label_column: Optional[str] = None, 
    time_column: Optional[str] = None,
    xai_settings: Optional[Dict[str, Any]] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> int: # Return 1 for success, 0 for failure
    """
    Runs a batch job including data simulation/import, model training, detection,
    evaluation, optional XAI, and logs summary statistics and metadata.
    """
    print(f"Starting Batch-job: {name}")
    sys.stdout.flush()

    # --- Timing & Initialization ---
    overall_start_time = time.perf_counter()
    sim_duration, training_duration, detection_duration, xai_duration = 0, 0, 0, 0
    sim_end_time, train_end_time, detect_end_time, xai_end_time = None, None, None, None
    evaluation_results = {}
    detection_results = {}
    run_status = "Failed" # Default status
    run_summary = {} # Initialize summary dict
    df = pd.DataFrame() # Initialize df
    training_data = pd.DataFrame()
    testing_data = pd.DataFrame()
    feature_columns = []
    anomaly_rows = pd.DataFrame()
    df_all_data_eval = pd.DataFrame()
    df_eval = pd.DataFrame()
    actual_label_col = label_column or 'label' # Use provided or default 'label'
    sequence_length = None # Initialize
    timestamp_col_name = 'timestamp'
    xai_runner_instance = None

    try:
        # --- Simulation / Data Import ---
        sim_start_time = time.perf_counter() # Start sim timer here
        actual_inj_params_for_xairunner = None # For passing to XAIRunner
        if inj_params is not None:            
            processed_anomaly_settings_for_sim = []
            actual_inj_params_for_xairunner = inj_params # Pass the raw structure
            
            # inj_params is a list of lists of dictionaries, e.g., [[{...}, {...}], [{...}]]
            for anomaly_group in inj_params: # Iterate through each group of anomalies
                if not isinstance(anomaly_group, list):
                    print(f"WARNING: Expected a list for an anomaly group, but got {type(anomaly_group)}. Skipping this group: {anomaly_group}")
                    continue
                for params_dict in anomaly_group: # Iterate through each anomaly definition (dict) in the group
                    if not isinstance(params_dict, dict):
                        print(f"WARNING: Expected a dictionary for anomaly parameters, but got {type(params_dict)}. Skipping this entry: {params_dict}")
                        continue

                    # Safely get parameters from the dictionary
                    anomaly_type = params_dict.get("anomaly_type")
                    timestamp_str = params_dict.get("timestamp")
                    magnitude_str = params_dict.get("magnitude")
                    percentage_str = params_dict.get("percentage")
                    columns = params_dict.get("columns") # Should be a list, e.g., ['V7']
                    duration = params_dict.get("duration") # String, e.g., '10s'

                    # Validate required parameters and convert types
                    missing_keys = []
                    if anomaly_type is None: missing_keys.append("anomaly_type")
                    if timestamp_str is None: missing_keys.append("timestamp")
                    if magnitude_str is None: missing_keys.append("magnitude")
                    if percentage_str is None: missing_keys.append("percentage")
                    # columns and duration can be optional depending on AnomalySetting definition

                    if missing_keys:
                        print(f"WARNING: Missing required keys {missing_keys} in anomaly params: {params_dict}. Skipping.")
                        continue
                    
                    try:
                        timestamp = int(timestamp_str)
                        magnitude = int(magnitude_str)
                        percentage = int(percentage_str)
                    except ValueError as ve:
                        print(f"WARNING: Error converting string to int for anomaly params: {params_dict}. Error: {ve}. Skipping.")
                        continue
                    except TypeError as te: # Handles if any _str is None due to missing key not caught above
                        print(f"WARNING: Non-string type encountered during int conversion for anomaly params: {params_dict}. Error: {te}. Skipping.")
                        continue

                    # Assuming AnomalySetting expects columns as a list and duration as a string.
                    # Add validation for columns if it's expected to be a list.
                    if columns is not None and not isinstance(columns, list):
                        print(f"WARNING: 'columns' parameter expected to be a list, but got {type(columns)}: {columns}. Using as is or skipping.")
                        # Decide how to handle: skip, or try to use, or wrap in a list if appropriate
                        # For now, let's assume AnomalySetting can handle it or it's an error to be skipped.
                        # If columns must be a list:
                        # columns = [columns] if not isinstance(columns, list) else columns

                    current_anomaly_setting = AnomalySetting(
                        anomaly_type, timestamp, magnitude,
                        percentage, columns, duration
                    )
                    processed_anomaly_settings_for_sim.append(current_anomaly_setting)
            
            # This line assumes 'path', 'name', 'debug' are defined elsewhere in your function
            batch_job = Job(filepath=path, anomaly_settings=processed_anomaly_settings_for_sim, simulation_type="batch", speedup=None, table_name=name, debug=debug)
            # print(f"Successfully processed {len(processed_anomaly_settings_for_sim)} anomaly settings.") # For feedback
        else:
            batch_job = Job(filepath=path, simulation_type="batch", anomaly_settings=None, speedup=None, table_name=name, debug=debug)

        sim_engine = se()
        # Pass the actual label column name to the simulator if it uses it
        result = sim_engine.main(db_conn_params=db_conn_params, job=batch_job, timestamp_col_name=time_column, label_col_name=actual_label_col)
        sim_end_time = time.perf_counter()
        sim_duration = sim_end_time - sim_start_time # Use sim_start_time
        # print(f"Batch import/simulation took {sim_duration:.2f}s")
        actual_label_col = 'label'

        if result != 1:
            raise RuntimeError("Simulation engine did not complete successfully.")

        # --- Read Data from DB ---
        api = TimescaleDBAPI(db_conn_params)
        # print(f"Reading data for table/job: {name}")
        df = api.read_data(datetime.fromtimestamp(0), name)
        if df.empty: raise ValueError("DataFrame read from DB is empty.")
        if PRIMARY_KEY_COLUMN not in df.columns:
            raise ValueError(f"Primary key column '{PRIMARY_KEY_COLUMN}' not found in DataFrame read from table '{name}'.")
        # print(f"DEBUG: Columns read from DB for job '{name}': {df.columns.tolist()}")
        # Ensure timestamp column exists and convert if necessary BEFORE splitting
        if timestamp_col_name and timestamp_col_name in df.columns:
            # Assuming read_data returns timezone-aware timestamps if applicable
            df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name])
            # print(f"Timestamp column '{timestamp_col_name}' read and converted.")
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            timestamp_col_name = 'timestamp' # Use default if found
            # print("Default 'timestamp' column read and converted.")
        else:
            warnings.warn("Timestamp column not found or specified. Time-based operations might fail.", RuntimeWarning)
            # Handle case where timestamp is missing - maybe default to index?

        # --- Data Splitting ---
        training_data, testing_data = split_data(df)
        if training_data.empty or testing_data.empty:
            warnings.warn("Training or testing data split resulted in empty DataFrame.", RuntimeWarning)
            # Handle fallback if needed

        # --- Feature Column Definition ---
        # Exclude ID, timestamp, labels, and existing flag columns from features
        cols_to_exclude = {PRIMARY_KEY_COLUMN, timestamp_col_name, actual_label_col, 'injected_anomaly', 'is_anomaly'}
        potential_feature_cols = [col for col in df.columns 
                                  if col not in cols_to_exclude and not pd.api.types.is_datetime64_any_dtype(df[col])]
        if not potential_feature_cols:
             raise ValueError("Could not identify any feature columns after exclusion.")
        feature_columns = potential_feature_cols
        # print(f"Identified Features ({len(feature_columns)}): {feature_columns}")

        # Use defensive selection in case columns were dropped
        training_features_df = training_data[[col for col in feature_columns if col in training_data.columns]]
        testing_features_df = testing_data[[col for col in feature_columns if col in testing_data.columns]]
        all_features_df = df[[col for col in feature_columns if col in df.columns]] # For detection

        # --- Impute NaNs ---
        # Now you can safely modify these copies
        # print(f"DEBUG run_batch: Imputing NaNs in training_features_df (shape: {training_features_df.shape}) using mean.")
        for col in training_features_df.columns:
            if training_features_df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(training_features_df[col]):
                    # Use .loc to ensure modification of the DataFrame directly
                    training_features_df.loc[:, col] = training_features_df[col].fillna(training_features_df[col].mean())
                else:
                    training_features_df.loc[:, col] = training_features_df[col].fillna(training_features_df[col].mode()[0] if not training_features_df[col].mode().empty else 'missing')

        # print(f"DEBUG run_batch: Imputing NaNs in all_features_df (shape: {all_features_df.shape}) using mean.")
        for col in all_features_df.columns:
            if all_features_df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(all_features_df[col]):
                    all_features_df.loc[:, col] = all_features_df[col].fillna(all_features_df[col].mean()) # Use mean from testing_features_df
                else:
                    all_features_df.loc[:, col] = all_features_df[col].fillna(all_features_df[col].mode()[0] if not all_features_df[col].mode().empty else 'missing')
        
        # print(f"DEBUG run_batch: Imputing NaNs in testing_features_df (shape: {testing_features_df.shape}) using mean.")
        for col in testing_features_df.columns:
            if testing_features_df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(testing_features_df[col]):
                    testing_features_df.loc[:, col] = testing_features_df[col].fillna(testing_features_df[col].mean()) # Use mean from testing_features_df
                else:
                    testing_features_df.loc[:, col] = testing_features_df[col].fillna(testing_features_df[col].mode()[0] if not testing_features_df[col].mode().empty else 'missing')

        # print(f"DEBUG run_batch: Imputing NaNs in testing_features_df (shape: {testing_features_df.shape}) using mean.")
        for col in testing_data.columns:
            if testing_data[col].isnull().any():
                if pd.api.types.is_numeric_dtype(testing_data[col]):
                    testing_data.loc[:, col] = testing_data[col].fillna(testing_data[col].mean()) # Use mean from testing_features_df
                else:
                    testing_data.loc[:, col] = testing_data[col].fillna(testing_data[col].mode()[0] if not testing_data[col].mode().empty else 'missing')

        # --- Anomaly Row Extraction (Ground Truth) ---
        if actual_label_col in df.columns:
            anomaly_rows = get_anomaly_rows(df, label_column=actual_label_col, anomaly_value=1)
            # print(f"Found {len(anomaly_rows)} ground truth anomaly rows using label '{actual_label_col}'.")
        else:
            print(f"Warning: Label column '{actual_label_col}' not found. Cannot extract ground truth anomalies.")

        # --- Model Training ---
        training_start_time = time.perf_counter()
        effective_model_params = model_params or {}
        # print(f"Getting model '{model}' with parameters: {effective_model_params}")
        model_instance = get_model(model, **effective_model_params) # Assumes get_model handles params

        model_name_lower = model.lower()

        if model_name_lower in UNSUPERVISED_MODELS:
            # --- Unsupervised Model ---
            # print(f"Model '{model}' identified as unsupervised. Training on features only.")
            if training_features_df.empty:
                raise ValueError("Training features DataFrame is empty, cannot train unsupervised model.")
            # print(f"Calling model.run with features-only data. Shape: {training_features_df.shape}")
            # Pass only features to the unsupervised model's run method
            model_instance.run(training_features_df)
        else:
            # --- Supervised Model (or unknown, default to supervised) ---
            # print(f"Model '{model}' identified as supervised/unknown. Training with features and label.")
            if training_data.empty:
                raise ValueError("Training data DataFrame is empty for supervised model.")
            if actual_label_col not in training_data.columns:
                raise ValueError(f"Label column '{actual_label_col}' missing from training_data. Columns: {training_data.columns.tolist()}")

            # Define the exact columns needed by the supervised model internally
            columns_for_supervised_training = feature_columns + [actual_label_col] # List of feature names + the label name

            # Select only these specific columns from the training_data slice
            # Ensure these columns actually exist in training_data to avoid KeyErrors
            valid_cols_for_training = [col for col in columns_for_supervised_training if col in training_data.columns]
            if len(valid_cols_for_training) != len(columns_for_supervised_training):
                missing_cols = set(columns_for_supervised_training) - set(valid_cols_for_training)
                # Raise error because model likely depends on these specific columns
                raise ValueError(f"Columns {missing_cols} required for supervised training not found in training_data slice.")

            # Create the specific DataFrame to pass to the model
            training_data_for_model = training_data[valid_cols_for_training]

            if training_data_for_model.empty:
                raise ValueError("DataFrame prepared for supervised model training (features + label) is empty.")

            # print(f"Calling model.run with training data containing ONLY specific features and label '{actual_label_col}'. Shape: {training_data_for_model.shape}")
            # Pass the filtered DataFrame (only features + label) to the model's run method
            model_instance.run(training_data_for_model)

        train_end_time = time.perf_counter()
        training_duration = train_end_time - training_start_time
        # print(f"Training took {training_duration:.2f}s")

        # --- Sequence Length ---
        sequence_length = getattr(model_instance, 'sequence_length', 1) # Default to 1 if not found
        if not isinstance(sequence_length, int) or sequence_length <= 0: sequence_length = 1
        # print(f"Using sequence_length: {sequence_length}")

        # --- Anomaly Detection ---
        detect_start_time = time.perf_counter()
        # Ensure detection data is not empty
        if all_features_df.empty:
            raise ValueError("Features DataFrame for detection is empty.")
        evaluation = model_instance.detect(testing_features_df)
        res = model_instance.detect(all_features_df)
        detect_end_time = time.perf_counter()
        detection_duration = detect_end_time - detect_start_time
        # print(f"Anomaly detection took {detection_duration:.2f}s. Results length: {len(res)}")

        # --- Assign Results & Evaluate ---
        # Prepare df_eval with original index and label for evaluation
        df_eval = testing_data.copy() # Copy only needed cols initially
        df_eval['is_anomaly'] = False # Initialize column
        df_all_data_eval = df.copy()
        df_all_data_eval['is_anomaly'] = False

        expected_padding = sequence_length - 1
        if len(res) == len(df_all_data_eval):
            # print("Assigning detection results directly.")
            df_all_data_eval['is_anomaly'] = res.values if isinstance(res, pd.Series) else res
        elif len(res) == len(df_all_data_eval) - expected_padding and expected_padding >= 0 :
            # print(f"Padding detection results with {expected_padding} 'False' values at the beginning.")
            padding = [False] * expected_padding
            res_list = list(res.values) if isinstance(res, pd.Series) else list(res)
            df_all_data_eval['is_anomaly'] = padding + res_list
        else:
            # Handle unexpected length difference more robustly
            warnings.warn(f"Unexpected length difference: results ({len(res)}), df ({len(df_all_data_eval)}), expected padding ({expected_padding}). Check model's detect output and sequence_length.", RuntimeWarning)
            # Attempt assignment if possible, otherwise evaluation might fail
            min_len = min(len(res), len(df_all_data_eval))
            df_all_data_eval['is_anomaly'][-min_len:] = list(res)[-min_len:] # Try aligning from end

        expected_padding = sequence_length - 1
        if len(evaluation) == len(df_eval):
            # print("Assigning detection results directly.")
            df_eval['is_anomaly'] = evaluation.values if isinstance(evaluation, pd.Series) else evaluation
        elif len(evaluation) == len(df_eval) - expected_padding and expected_padding >= 0 :
            # print(f"Padding detection results with {expected_padding} 'False' values at the beginning.")
            padding = [False] * expected_padding
            evaluation_list = list(evaluation.values) if isinstance(evaluation, pd.Series) else list(evaluation)
            df_eval['is_anomaly'] = padding + evaluation_list
        else:
            # Handle unexpected length difference more robustly
            warnings.warn(f"Unexpected length difference: results ({len(evaluation)}), df ({len(df_eval)}), expected padding ({expected_padding}). Check model's detect output and sequence_length.", RuntimeWarning)
            # Attempt assignment if possible, otherwise evaluation might fail
            min_len = min(len(evaluation), len(df_eval))
            df_eval['is_anomaly'][-min_len:] = list(evaluation)[-min_len:] # Try aligning from end

        # Update anomalies in DB (using original timestamps)
        predicted_anomalies_mask = df_all_data_eval['is_anomaly'] == True
        # Select rows from the original 'df' (which includes the PRIMARY_KEY_COLUMN)
        anomaly_df_for_update = df.loc[predicted_anomalies_mask & df[PRIMARY_KEY_COLUMN].notna()]
        
        # print(f'Found {len(anomaly_df_for_update)} anomalies to update in DB via PK.')
        if not anomaly_df_for_update.empty:
            anomaly_pk_values = anomaly_df_for_update[PRIMARY_KEY_COLUMN].tolist()
            if anomaly_pk_values:
                # Call the new DB API function that updates by primary key
                # This method needs to be implemented in your TimescaleDBAPI class.
                api.update_anomalies(table_name=name,
                                    pk_column_name=PRIMARY_KEY_COLUMN,
                                    anomaly_pk_values=anomaly_pk_values)
            else:
                print("No valid primary key values found for the detected anomalies.")
        else:
            print("No anomalies detected in current run, or no anomalies with valid PKs.")

        detection_results = evaluate_classification(df_all_data_eval) # Evaluate using actual and predicted labels
        evaluation_results = evaluate_classification(df_eval) # Evaluate the testing set
        # print("Evaluation Results:", evaluation_results)

        # --- XAI Execution (Conditional) ---
        xai_start_time = time.perf_counter() # Start XAI timer here
        model_wrapper = None # Initialize
        avg_ndcg_scores = {} # Initialize NDCG results dict
        
        interpretation_list = ['lstm', 'XGBoost', 'decision_tree'] # higher_is_anomaly
        
        if xai_settings and isinstance(xai_settings, dict):
            interpretation = 'higher_is_anomaly' # Default
            # Determine interpretation based on model name string
            if model.lower() not in interpretation_list: interpretation = 'lower_is_anomaly'
            # Add other model types here
            else: warnings.warn(f"Unknown model type '{model}' for score interpretation. Assuming higher score is anomaly.", RuntimeWarning)

            try:
                # print("Wrapping model for XAI...")
                model_wrapper = ModelWrapperForXAI(
                    actual_model_instance=model_instance,
                    feature_names=feature_columns,
                    score_interpretation=interpretation
                )
                # print(f"Model wrapped. Interpretation: '{interpretation}'")

                # print("Instantiating XAIRunner...")
                xai_runner_instance = XAIRunner(
                    xai_settings=xai_settings,
                    model_wrapper=model_wrapper,
                    sequence_length=sequence_length,
                    feature_columns=feature_columns,
                    actual_label_col=actual_label_col,
                    # Use training_features_df for continuous features list (adjust if needed)
                    continuous_features_list=training_features_df.columns.tolist(),
                    job_name=name,
                    mode='classification',
                    output_dir=OUTPUT_DIR, # Use defined output dir
                    # --- Pass params for NDCG ---
                    inj_params=actual_inj_params_for_xairunner, # Pass the original inj_params
                    timestamp_col_name=timestamp_col_name
                )

                # print("Running XAI explanations...")
                # Prepare dataframes needed by XAIRunner
                # Assuming training_data/testing_data include labels and timestamps if needed by XAIRunner internals
                # Choose the data to explain (e.g., testing data, all data, or specific anomalies)
                data_source_for_exp = df.copy() # Make a copy to be safe
                if timestamp_col_name in data_source_for_exp.columns and \
                   not pd.api.types.is_datetime64_any_dtype(data_source_for_exp[timestamp_col_name]):
                    # print(f"DEBUG run_batch: Converting data_source_for_exp['{timestamp_col_name}'] to datetime again.")
                    data_source_for_exp[timestamp_col_name] = pd.to_datetime(data_source_for_exp[timestamp_col_name])
                
                # print(f"DEBUG run_batch: data_source_for_exp for XAIRunner: shape={data_source_for_exp.shape}")
                # print(f"DEBUG run_batch: data_source_for_exp '{timestamp_col_name}' MIN: {data_source_for_exp[timestamp_col_name].min() if timestamp_col_name in data_source_for_exp else 'N/A'}")
                # print(f"DEBUG run_batch: data_source_for_exp '{timestamp_col_name}' MAX: {data_source_for_exp[timestamp_col_name].max() if timestamp_col_name in data_source_for_exp else 'N/A'}")
                
                xai_runner_instance.run_explanations(
                    training_features_df=training_features_df,
                    training_df_with_labels=training_data,
                    data_source_for_explanation=data_source_for_exp # df used here
                )
                # print("XAI execution completed.")

                # --- Collect NDCG Results ---
                if xai_runner_instance.ndcg_results:
                    # print("Aggregating NDCG results...")
                    for method, k_map in xai_runner_instance.ndcg_results.items():
                        avg_ndcg_scores.setdefault(method, {})
                        for k, scores_list in k_map.items():
                            if scores_list: # Ensure not empty
                                avg_ndcg_scores[method][f"NDCG@{k}"] = np.mean(scores_list)
                            else:
                                avg_ndcg_scores[method][f"NDCG@{k}"] = 0.0 # Or None/NaN
                    # print(f"Average NDCG Scores: {avg_ndcg_scores}")


            except Exception as xai_err:
                print(f"ERROR during XAI setup or execution: {xai_err}")
                traceback.print_exc()
            
            xai_end_time = time.perf_counter()
            xai_duration = xai_end_time - xai_start_time
        else:
            # print("Skipping XAI (no settings provided or error during setup).")
            xai_end_time = detect_end_time 

        run_status = "Success"

    except Exception as e:
        print(f"An error occurred during run_batch for job '{name}': {e}")
        traceback.print_exc()
        run_status = "Failed" # Ensure status reflects failure

    finally:
        overall_end_time = time.perf_counter()
        overall_duration = overall_end_time - overall_start_time
        # print(f"Total run_batch execution for job '{name}' took {overall_duration:.2f}s. Status: {run_status}")

        # --- Gather Summary Data ---
        # Calculate additional metrics if evaluation succeeded
        if evaluation_results:
            tp = evaluation_results.get("correct_anomalies", 0)
            fp = evaluation_results.get("false_positives", 0)
            fn = evaluation_results.get("false_negatives", 0)
            tn = evaluation_results.get("correct_non_anomalies", 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            evaluation_results["precision"] = round(precision, 4)
            evaluation_results["recall_tpr"] = round(recall, 4)
            evaluation_results["f1_score"] = round(f1_score, 4)
            evaluation_results["specificity_tnr"] = round(specificity, 4)
            
        if detection_results:
            tp = detection_results.get("correct_anomalies", 0)
            fp = detection_results.get("false_positives", 0)
            fn = detection_results.get("false_negatives", 0)
            tn = detection_results.get("correct_non_anomalies", 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            detection_results["precision"] = round(precision, 4)
            detection_results["recall_tpr"] = round(recall, 4)
            detection_results["f1_score"] = round(f1_score, 4)
            detection_results["specificity_tnr"] = round(specificity, 4)
            
        cv_metrics = {}
        if model_instance is not None: # Check if model_instance was created
            if hasattr(model_instance, 'get_validation_scores'):
                try:
                    cv_metrics = model_instance.get_validation_scores()
                    if not isinstance(cv_metrics, dict): # Ensure it's a dict
                        print(f"Warning: get_validation_scores() did not return a dict, got {type(cv_metrics)}. Resetting cv_metrics.")
                        cv_metrics = {}
                except Exception as cv_err:
                    print(f"Warning: Error calling get_validation_scores(): {cv_err}")
                    cv_metrics = {} # Default to empty dict on error

            if hasattr(model_instance, 'avg_best_iteration_cv_') and model_instance.avg_best_iteration_cv_ is not None:
                cv_metrics['avg_best_iteration_cv'] = model_instance.avg_best_iteration_cv_
            if hasattr(model_instance, 'avg_best_score_cv_') and model_instance.avg_best_score_cv_ is not None:
                cv_metrics['avg_best_score_cv_from_xgb_eval_metric'] = model_instance.avg_best_score_cv_
        else:
            print("Warning: model_instance is None, cannot retrieve CV metrics.")
            
        # Add avg_ndcg_scores to run_summary
        xai_eval_metrics = {}
        if avg_ndcg_scores:
             xai_eval_metrics["ndcg_scores"] = avg_ndcg_scores
             
        # Add individual XAI method timings
        individual_xai_timings = {}
        if xai_runner_instance: # Check if XAIRunner was instantiated
            individual_xai_timings = xai_runner_instance.get_xai_method_timings()

        run_summary = {
            "job_name": name,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": run_status,
            "dataset_path": path,
            "model_name": model,
            "model_params": model_params or {},
            "xai_settings": xai_settings or {},
            "xai_evaluation_metrics": xai_eval_metrics, # NDCG
            "anomaly_injection_params": inj_params or [],
            "label_column_used": actual_label_col,
            "data_total_rows": len(df),
            "data_training_rows": len(training_data),
            "data_testing_rows": len(testing_data),
            "data_num_features": len(feature_columns),
            "data_num_anomalies_ground_truth": len(anomaly_rows),
            "data_num_anomalies_predicted": int(df_eval["is_anomaly"].sum()) if 'is_anomaly' in df_eval.columns else 0,
            "sequence_length": sequence_length,
            "evaluation_metrics": evaluation_results,
            "all_data_evaluation_metrics": detection_results,
            "cross_validation_metrics": cv_metrics,
            "execution_time_total_seconds": round(overall_duration, 4),
            "execution_time_simulation_seconds": round(sim_duration, 4),
            "execution_time_training_seconds": round(training_duration, 4),
            "execution_time_detection_seconds": round(detection_duration, 4),
            "execution_time_xai_seconds": round(xai_duration, 4),
        }
        
        # Add individual XAI method timings to the summary
        for method, timing in individual_xai_timings.items():
            run_summary[f"execution_time_xai_{method}_seconds"] = timing

        # --- Save Summary ---
        save_run_summary(run_summary, name, OUTPUT_DIR)

        sys.stdout.flush() # Ensure all print statements are flushed

        return 1 if run_status == "Success" else 0


# Starts processing of dataset as a stream
def run_stream(db_conn_params, 
            model: str, 
            path: str, 
            name: str, 
            speedup: int, 
            inj_params: dict=None, 
            debug=False) -> None:
    print("Starting Stream-job!")
    sys.stdout.flush()

    if inj_params is not None:
        anomaly_settings = []  # Create a list to hold AnomalySetting objects
        for params in inj_params:  # Iterate over the list of anomaly dictionaries
            anomaly = AnomalySetting(
                params.get("anomaly_type", None),
                int(params.get("timestamp", None)),
                int(params.get("magnitude", None)),
                int(params.get("percentage", None)),
                params.get("columns", None),
                params.get("duration", None)
            )
            anomaly_settings.append(anomaly)  # Add the AnomalySetting object to the list
        stream_job = Job(filepath=path, anomaly_settings=anomaly_settings, simulation_type="stream", speedup=speedup, table_name=name, debug=debug)
    else:
        print("Should not inject anomaly.")
        stream_job = Job(filepath=path, simulation_type="stream", speedup=speedup, table_name=name, debug=debug)

    sim_engine = se()
    sim_engine.main(db_conn_params, stream_job)

def single_point_detection(api, simulation_thread, model, name, path):
    
    model_instance = get_model(model)
    df = pd.read_csv(path)
    model_instance.run(df)

    while not api.table_exists(name):
        time.sleep(1)
    
    
    timestamp = datetime.fromtimestamp(0)
    
    while simulation_thread.is_alive():
        df = api.read_data(datetime.fromtimestamp(0), name)
        timestamp = df["timestamp"].iloc[-1].to_pydatetime()
        print(df["timestamp"].iloc[-1])

        df["timestamp"] = df["timestamp"].apply(map_to_timestamp)
        df["timestamp"] = df["timestamp"].astype(float)

        res = model_instance.detect(df.iloc[:, :-2])
        df["is_anomaly"] = res
        
        anomaly_df = df[df["is_anomaly"] == True]
        arr = [datetime.fromtimestamp(timestamp) for timestamp in anomaly_df["timestamp"]]
        arr = [f'\'{str(time)}+00\'' for time in arr]
        
        api.update_anomalies(name, arr)
    
        time.sleep(1)


# Returns a list of models implemented in MODEL_DIRECTORY
def get_models() -> list:
    models = []
    for path in os.listdir(MODEL_DIRECTORY):
        file_path = MODEL_DIRECTORY + "/" + path
        if os.path.isfile(file_path):
            model_name = path.split(".")[0]
            models.append(model_name)

    # Removing the __init__, setup files and the .env file
    models.remove("")
    models.remove("model_interface")
    models.remove("__init__")
    models.remove("setup")
    models.remove("get_model")
    models.remove("model_wrapper")
    
    return models

# Returns a list of XAI mthods implemented in XAI_METHOD_DIRECTORY
def get_xai_methods() -> list:
    methods = []
    for path in os.listdir(XAI_METHOD_DIRECTORY):
        if os.path.isfile(os.path.join(XAI_METHOD_DIRECTORY, path)):
            method_name = path.split(".")[0]
            methods.append(method_name)

    # Removing the __init__, setup files and the .env file
    methods.remove("__init__")

    return methods

# Returns a list of injection methods implemented in INJECTION_METHOD_DIRECTORY
def get_injection_methods() -> list:
    injection_methods = []

    for path in os.listdir(INJECTION_METHOD_DIRECTORY):
        if os.path.isfile(os.path.join(INJECTION_METHOD_DIRECTORY, path)):
            method_name = path.split(".")[0]
            injection_methods.append(method_name)

    injection_methods.remove("__init__")
    return injection_methods

# Fetching datasets from the dataset directory
def get_datasets() -> list:
    datasets = []
    for path in os.listdir(DATASET_DIRECTORY):
        file_path = DATASET_DIRECTORY + "/" + path
        if os.path.isfile(file_path):
            dataset = path
            datasets.append(dataset)

    return datasets

# Gets content of complete file to the backend
def import_dataset(conn: socket, path: str, timestamp_column: str) -> None:
    file = open(path, "w")
    data = conn.recv(1024).decode("utf-8")
    while data:
        file.write(data)
        data = conn.recv(1024).decode("utf-8")
    file.close()
    
    # Change the timestamp column name to timestamp and move it to the first column
    df = pd.read_csv(path)
    df.rename(columns={timestamp_column: "timestamp"}, inplace=True)
    cols = df.columns.tolist()
    cols.remove("timestamp")
    cols = ["timestamp"] + cols
    df = df[cols]
    df.to_csv(path, index=False)