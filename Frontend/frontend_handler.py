import sys
import os
import traceback
import pandas as pd
import json
from io import StringIO

from api import BackendAPI

class FrontendHandler:
    def __init__(self, host, port):
        self.api = BackendAPI(host, port)

    def check_name(self, job_name) -> str:

        try:
            response = self.handle_get_all_jobs()
            if response is None: # Handle potential None from failed call
                 return "error-checking-name"
            if job_name in response:
                return "name-error"
            return "success"
        except Exception as e:
            print(f"Error in check_name: {e}")
            return "error-checking-name"


    def handle_run_batch(self, selected_dataset, selected_model, job_name, inj_params: dict=None, label_column=None, time_column=None, xai_params=None, model_params=None) -> str:

        response = self.check_name(job_name)
        if response == "success":
            try:
                if inj_params is None:
                     self.api.run_batch(selected_model, selected_dataset, job_name, inj_params=None, label_column=label_column, time_column=time_column,xai_params=xai_params, model_params=model_params)
                else:
                    self.api.run_batch(selected_model, selected_dataset, job_name, inj_params=[inj_params], label_column=label_column, time_column=time_column, xai_params=xai_params, model_params=model_params)
            except Exception as e:
                 print(f"Error calling self.api.run_batch: {e}")
                 return f"Error starting batch job: {e}"
        return response

    def handle_run_stream(self, selected_dataset, selected_model, job_name, speedup, inj_params: dict=None, label_column=None, xai_params=None) -> str:
         response = self.check_name(job_name)
         if response == "success":
             try:
                  self.api.run_stream(selected_model, selected_dataset, job_name, speedup, inj_params=[inj_params], label_column=label_column, xai_params=xai_params, model_params=None)
             except Exception as e:
                  print(f"Error calling self.api.run_stream: {e}")
                  return f"Error starting stream job: {e}"
         return response

    def handle_get_data(self, timestamp, job_name):
        """
        Handles fetching data from the backend API for a specific job.
        Returns an empty DataFrame if the backend call fails or returns invalid data.
        """
        print(f"FrontendHandler: Attempting to get data for job '{job_name}' from backend...")
        data = None # Initialize data to None
        try:
            # Call the backend API 
            data = self.api.get_data(timestamp, job_name)

            # --- Check if data received from backend is valid ---
            if data is None:
                print(f"Warning: Received None from self.api.get_data for job '{job_name}'. Backend likely unreachable or returned error.")
                return pd.DataFrame() # Return empty DataFrame

            if not isinstance(data, dict):
                 print(f"Warning: Received non-dict data from self.api.get_data for job '{job_name}': {type(data)}. Returning empty DataFrame.")
                 return pd.DataFrame() # Return empty DataFrame

            if "data" not in data or data["data"] is None:
                print(f"Warning: Received dict from backend, but 'data' key is missing or None for job '{job_name}'. Response: {data}. Returning empty DataFrame.")
                return pd.DataFrame() # Return empty DataFrame

            # --- If data seems valid, proceed with parsing ---
            print(f"FrontendHandler: Received data structure with 'data' key. Attempting to parse JSON...")
            json_string = data["data"]
            df = pd.read_json(StringIO(json_string), orient="split")
            print(f"FrontendHandler: Successfully parsed data for job '{job_name}'. DataFrame shape: {df.shape}")
            return df

        except json.JSONDecodeError as json_err:
            # Catch errors during pd.read_json if backend sent invalid JSON
            print(f"Error: Failed to decode JSON received from backend for job '{job_name}': {json_err}")
            print(f"Received data content (first 100 chars): {str(data)[:100]}...") # Log snippet of bad data
            return pd.DataFrame() # Return empty DataFrame
        except Exception as e:
            # Catch any other unexpected errors during the process
            print(f"Error in handle_get_data for job '{job_name}': {e}")
            traceback.print_exc()
            return pd.DataFrame() # Return empty DataFrame

    def handle_get_running(self):
        try:
             response = self.api.get_running()
             if response is None:
                  print("Warning: Received None from self.api.get_running.")
                  # Return structure expected by callback on failure
                  return json.dumps({"running": []})
             return response
        except Exception as e:
             print(f"Error calling self.api.get_running: {e}")
             # Return structure expected by callback on failure
             return json.dumps({"running": []})


    def handle_cancel_job(self, job_name):
        response = self.check_name(job_name) 
        if response == "name-error": 
            try:
                self.api.cancel_job(job_name)
                return "success" # Indicate frontend call succeeded
            except Exception as e:
                 print(f"Error calling self.api.cancel_job for '{job_name}': {e}")
                 return f"Error cancelling job: {e}" # Return error message
        elif response == "success":
             return "Job not found."
        else:
            return "Error checking job status before cancelling."


    def handle_get_models(self):
        try:
            models_json = self.api.get_models()
            if models_json is None: raise ValueError("Received None from API")
            models = json.loads(models_json)
            return models.get("models", []) # Return empty list if key missing
        except Exception as e:
             print(f"Error getting/parsing models: {e}")
             return [] # Return empty list on error


    def handle_get_xai_methods(self):
        methods_list = []
        try:
            response_data = self.api.get_xai_methods()
            if response_data is None or response_data == "":
                 print("Warning: API get_xai_methods returned None or empty.")
                 return methods_list
            methods_data = json.loads(response_data)
            if isinstance(methods_data, dict) and 'methods' in methods_data and isinstance(methods_data['methods'], list):
                 methods_list = methods_data["methods"]
            else:
                 print(f"Warning: Unexpected JSON structure from get_xai_methods: {methods_data}")
        except json.JSONDecodeError as json_err: print(f"Error decoding JSON from get_xai_methods: {json_err} | Data: {response_data!r}")
        except ConnectionError as conn_err: print(f"Connection Error in get_xai_methods: {conn_err}")
        except Exception as e: print(f"Generic Error in handle_get_xai_methods: {e}"); traceback.print_exc()
        return methods_list


    def handle_get_injection_methods(self):
        try:
            methods_json = self.api.get_injection_methods()
            if methods_json is None: raise ValueError("Received None from API")
            injection_methods = json.loads(methods_json)
            return injection_methods.get("injection_methods", [])
        except Exception as e:
             print(f"Error getting/parsing injection methods: {e}")
             return []


    def handle_get_datasets(self):
        try:
            datasets_json = self.api.get_datasets()
            if datasets_json is None: raise ValueError("Received None from API")
            datasets = json.loads(datasets_json)
            return datasets.get("datasets", [])
        except Exception as e:
             print(f"Error getting/parsing datasets: {e}")
             return []


    def handle_import_dataset(self, file_path, timestamp_column: str):
        try:
             self.api.import_dataset(file_path, timestamp_column)
        except Exception as e:
             print(f"Error calling self.api.import_dataset: {e}")
             # Decide how to signal error - raise? return status?


    def handle_get_all_jobs(self):
        try:
            jobs_json = self.api.get_all_jobs()
            if jobs_json is None: raise ValueError("Received None from API")
            jobs = json.loads(jobs_json)
            return jobs.get("jobs", [])
        except Exception as e:
             print(f"Error getting/parsing all jobs: {e}")
             return [] # Return empty list


    def handle_get_columns(self, job_name):
        response = self.check_name(job_name) # check_name handles its errors
        if response == "name-error":
            try:
                columns_json = self.api.get_columns(job_name)
                if columns_json is None: raise ValueError("Received None from API")
                columns = json.loads(columns_json)
                return columns.get("columns", [])
            except Exception as e:
                 print(f"Error getting/parsing columns for job '{job_name}': {e}")
                 return "Error getting columns" # Return error message/status
        return response # Return 'success' (job not found) or 'error-checking-name'


    def handle_get_dataset_columns(self, dataset):
        if dataset == None: return []
        try:
            response = self.api.get_dataset_columns(dataset)
            if response is None: raise ValueError("Received None from API")
            columns = json.loads(response)
            return columns.get("columns", [])
        except Exception as e:
             print(f"Error getting/parsing dataset columns for '{dataset}': {e}")
             return []