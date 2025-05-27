import socket
import json
import sys
import threading
from time import sleep
import os
import traceback
import execute_calls
from timescaledb_api import TimescaleDBAPI
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
import shutil
from Simulator.FileFormats.read_csv import read_csv

load_dotenv()
BACKEND_HOST = os.getenv('BACKEND_HOST')
BACKEND_PORT = int(os.getenv('BACKEND_PORT'))
DATABASE = {
    "HOST": os.getenv('DATABASE_HOST'),
    "PORT": int(os.getenv('DATABASE_PORT')),
    "USER": os.getenv('DATABASE_USER'),
    "PASSWORD": os.getenv('DATABASE_PASSWORD'),
    "DATABASE": os.getenv('DATABASE_NAME')
}

DATASET_DIRECTORY = "./Datasets/"
OUTPUT_DIR = "/data"

backend_data = {
    "started-jobs": [],
    "running-jobs": []
}

def discover_existing_jobs(db_api: TimescaleDBAPI):
    print("Discovering existing job tables...")
    discovered_jobs = []
    try:
        all_tables = db_api.list_all_tables()

        prefix_batch = "job_batch_"
        prefix_stream = "job_stream_"

        for table_name in all_tables:
            job_type = None
            job_name_part = None

            if table_name.startswith(prefix_batch):
                job_type = "batch"
                job_name_part = table_name
            elif table_name.startswith(prefix_stream):
                job_type = "stream"
                job_name_part = table_name

            if job_type and job_name_part:
                # Check if this job name is already known (e.g., from a concurrent start?)
                is_known = any(j["name"] == job_name_part for j in backend_data["running-jobs"]) or \
                           any(j["name"] == job_name_part for j in backend_data["started-jobs"])

                if not is_known:
                    print(f"  Discovered existing job: {job_name_part} (Type: {job_type})")
                    job_info = {
                        "name": job_name_part,
                        "type": job_type,
                        "thread": None, # Explicitly None as the thread is lost
                        "discovered_at_startup": True
                    }
                    discovered_jobs.append(job_info)

    except Exception as e:
        print(f"Error discovering existing jobs: {e}")

    # Add discovered jobs to the list
    backend_data["running-jobs"].extend(discovered_jobs)
    print(f"Finished discovery. Total 'running' jobs now: {len(backend_data['running-jobs'])}")

def main():
    # Create a thread listening for requests
    listener_thread = threading.Thread(target=__request_listener)
    listener_thread.daemon = True

    db_conn_params = {
        "user": DATABASE["USER"],
        "password": DATABASE["PASSWORD"],
        "host": DATABASE["HOST"],
        "port": DATABASE["PORT"],
        "database": DATABASE["DATABASE"]
    }

    try:
        backend_data["db_api"] = TimescaleDBAPI(db_conn_params)
        print("Database API initialized.")

        # --- Run discovery AFTER db_api is initialized ---
        discover_existing_jobs(backend_data["db_api"])

        # --- Start the listener ---
        listener_thread.start()
        print("Request listener started.")

    except Exception as e:
        print(f"CRITICAL ERROR during initialization: {e}")
        sys.exit(1) # Exit if DB connection fails

    print("Main thread started...")
    # Main loop serving the backend logic
    try:
        while True:
            # --- Main loop logic needs adjustment ---
            current_started = list(backend_data["started-jobs"]) # Iterate over a copy
            for job in current_started:
                if job.get("discovered_at_startup"): # Skip discovered jobs in this check
                    continue

                if job["type"] == "batch":
                    # Check if the thread finished *if* it exists
                    if job["thread"] and not job["thread"].is_alive():
                        print(f"Batch job '{job['name']}' thread finished. Moving to running-jobs.")
                        backend_data["running-jobs"].append(job)
                        backend_data["started-jobs"].remove(job)
                    # If thread is None (shouldn't happen for non-discovered), handle error?
                elif job["type"] == "stream":
                    # Check if table exists for newly started stream jobs
                    if backend_data["db_api"].table_exists(f"job_stream_{job['name']}"):
                        print(f"Stream job '{job['name']}' table found. Moving to running-jobs.")
                        backend_data["running-jobs"].append(job)
                        backend_data["started-jobs"].remove(job)

            sleep(1)

    except KeyboardInterrupt:
        print("Exiting backend...")

# Listens for incoming requests and handles them through the __handle_api_call function
def __request_listener():
    try: 
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((BACKEND_HOST, BACKEND_PORT))
        sock.listen()
        sock.settimeout(1)
    except Exception as e:
        print(e)

    while True:
        try: 
            conn, addr = sock.accept()
            print(f'Connected to {addr}')
            recv_data = conn.recv(4096)
            recv_data = recv_data.decode("utf-8")
            recv_dict = json.loads(recv_data)
            print(f"Received request: {recv_dict}")
            __handle_api_call(conn, recv_dict)
        except Exception as e:
            if str(e) != "timed out":
                print(e)
            pass

# Handles the incoming requests and sends a response back to the client
def __handle_api_call(conn, data: dict) -> None:
    match data["METHOD"]:
        case "run-batch":
            model = data["model"]
            dataset_path = DATASET_DIRECTORY + data["dataset"]
            name = data["name"]
            debug = data["debug"]
            label_column = data.get("label_column", None)
            time_column = data.get("time_column", None)

            inj_params = data.get("inj_params", None)
            xai_params = data.get("xai_params", None)
            model_params = data.get("model_params", None)

            db_conn_params = {
                "user": DATABASE["USER"],
                "password": DATABASE["PASSWORD"],
                "host": DATABASE["HOST"],
                "port": DATABASE["PORT"],
                "database": DATABASE["DATABASE"]
            }
            
            new_thread = threading.Thread(
            target=execute_calls.run_batch,
                args=(
                    db_conn_params,
                    model,
                    dataset_path,
                    name,
                    inj_params,
                    debug,
                    label_column,
                    time_column,
                    xai_params,
                    model_params,
                )
            )
            new_thread.daemon = True
            new_thread.start()

            job = {
                "name": name,
                "type": "batch",
                "thread": new_thread
            }

            backend_data["started-jobs"].append(job)

        case "run-stream":
            model = data["model"]
            dataset_path = DATASET_DIRECTORY + data["dataset"]
            name = data["name"]
            speedup = data["speedup"]
            debug = data["debug"]
            label_column = data.get("label_column", None)

            inj_params = data.get("inj_params", None)
            xai_params = data.get("xai_params", None)
            model_params = data.get("model_params", None)
            
            db_conn_params = {
                "user": DATABASE["USER"],
                "password": DATABASE["PASSWORD"],
                "host": DATABASE["HOST"],
                "port": DATABASE["PORT"],
                "database": DATABASE["DATABASE"]
            }

            stream_thread = threading.Thread(
                target=execute_calls.run_stream,
                args=(
                    db_conn_params,
                    model,
                    dataset_path,
                    name,
                    speedup,
                    inj_params,
                    debug,
                    label_column,
                    xai_params,
                    model_params
                )
            )            

            stream_thread.daemon = True
            stream_thread.start()
            detection_thread = threading.Thread(target=execute_calls.single_point_detection, args=(backend_data["db_api"], new_thread, model, name, dataset_path))
            detection_thread.daemon = True
            detection_thread.start()

            job = {
                "name": name,
                "type": "stream",
                "thread": stream_thread
            }

            backend_data["started-jobs"].append(job)
            
        case "get-data":
            try:
                # --- Correctly parse the 'from_timestamp' ISO string ---
                from_dt = datetime.fromisoformat(data["from_timestamp"])
                print(f"Parsed from_timestamp: {from_dt}")

                to_dt = None
                if data["to_timestamp"] is not None:
                    # --- Correctly parse the 'to_timestamp' ISO string ---
                    to_dt = datetime.fromisoformat(data["to_timestamp"])
                    print(f"Parsed to_timestamp: {to_dt}")

                # Call the database function with the correct datetime objects
                if to_dt is None:
                    df = backend_data["db_api"].read_data(from_dt, data["job_name"])
                else:
                    df = backend_data["db_api"].read_data(from_dt, data["job_name"], to_dt)

                # --- The rest of your data processing ---
                # Check if DataFrame is empty before proceeding
                if df is not None and not df.empty:
                    # Assuming 'timestamp' column exists and needs mapping/conversion
                    if "timestamp" in df.columns:
                        df["timestamp"] = df["timestamp"].apply(execute_calls.map_to_timestamp)
                        df["timestamp"] = df["timestamp"].astype(float) # Be careful if map_to_timestamp doesn't return numbers

                    data_json = df.to_json(orient="split")
                    df_dict = {"data": data_json}
                else:
                    # Handle empty DataFrame case - send back empty data structure
                    print("Warning: db_api.read_data returned empty DataFrame.")
                    df_dict = {"data": None} # Send None in 'data' key as per frontend expectation

                df_json = json.dumps(df_dict)
                conn.sendall(bytes(df_json, encoding="utf-8"))
                print("Data sent")

            except (ValueError, TypeError) as e:
                # Catch errors during timestamp parsing (e.g., invalid format)
                print(f"Error processing get-data request: Invalid timestamp format? {e}")
                # Send an error response back to the client
                error_dict = {"error": f"Invalid timestamp format: {e}", "data": None}
                error_json = json.dumps(error_dict)
                try:
                    conn.sendall(bytes(error_json, encoding="utf-8"))
                except Exception as send_err:
                    print(f"Failed to send error response: {send_err}")

            except Exception as e:
                # Catch any other unexpected errors during processing
                print(f"Error processing get-data request: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                # Send an error response back to the client
                error_dict = {"error": f"Internal server error: {e}", "data": None}
                error_json = json.dumps(error_dict)
                try:
                    conn.sendall(bytes(error_json, encoding="utf-8"))
                except Exception as send_err:
                    print(f"Failed to send error response: {send_err}")
        case "get-running":
            jobs = []
            for job in backend_data["running-jobs"]:
                new_job = {
                    "name": job["name"],
                    "type": job["type"]
                }
                jobs.append(new_job)
            running_dict = {
                "running": jobs
            }
            running_json = json.dumps(running_dict)
            conn.sendall(bytes(running_json, encoding="utf-8"))
        case "cancel-job":
            __cancel_job(data["job_name"])
        case "get-models":
            models = execute_calls.get_models()
            models_dict = {
                                "models": models
                            }
            models_json = json.dumps(models_dict)
            conn.sendall(bytes(models_json, encoding="utf-8"))
        case "get-xai-methods":
            methods = execute_calls.get_xai_methods()
            print(f"sending columns: {methods}")
            methods_dict = {
                                "methods": methods
                            }
            methods_json = json.dumps(methods_dict)
            conn.sendall(bytes(methods_json, encoding="utf-8"))
        case "get-injection-methods":
            injection_methods = execute_calls.get_injection_methods()
            injection_methods_dict = {
                                "injection_methods": injection_methods
                            }
            injection_methods_json = json.dumps(injection_methods_dict)
            conn.sendall(bytes(injection_methods_json, encoding="utf-8"))
        case "get-datasets":
            datasets = execute_calls.get_datasets()
            datasets_dict = {
                                "datasets": datasets
                            }
            datasets_json = json.dumps(datasets_dict)
            conn.sendall(bytes(datasets_json, encoding="utf-8"))
        case "import-dataset":
            path = DATASET_DIRECTORY + data["name"]
            conn.settimeout(1)
            # If the file does not exist, read the file contents written to the socket
            if not os.path.isfile(path):
                execute_calls.import_dataset(conn, path, data["timestamp_column"])
            # If the file already exists, empty the socket buffer and do nothing
            else:
                data = conn.recv(4096)
                while data:
                    data = conn.recv(4096)
        case "get-all-jobs":
            job_names = []

            for job in backend_data["running-jobs"]:
                job_names.append(job["name"])
            
            for job in backend_data["started-jobs"]:
                job_names.append(job["name"])
            
            jobs_dict = {
                            "jobs": job_names
                        }
            jobs_json = json.dumps(jobs_dict)
            conn.sendall(bytes(jobs_json, encoding="utf-8"))
        case "get-columns":
            columns = backend_data["db_api"].get_columns(data["name"])
            columns_dict = {
                                "columns": columns
                            }
            columns_json = json.dumps(columns_dict)
            conn.sendall(bytes(columns_json, encoding="utf-8"))
        case "get-dataset-columns":
            file_reader = read_csv(DATASET_DIRECTORY + data["dataset"])
            columns = file_reader.get_columns()
            print(f"sending columns: {columns}")
            columns_dict = {
                                "columns": columns
                            }
            columns_json = json.dumps(columns_dict)
            conn.sendall(bytes(columns_json, encoding="utf-8"))
        case _: 
            response_json = json.dumps({"error": "method-error-response" })
            conn.sendall(bytes(response_json, encoding="utf-8"))      
    conn.shutdown(socket.SHUT_RDWR)
    conn.close()
            
def __cancel_job(job_name: str) -> None:
    print("Cancelling job...")
    job_found = False
    for job in backend_data["running-jobs"]:
        if job["name"] == job_name:
            job_found = True
            print(f"Found job '{job_name}'. Proceeding with cancellation.")
            # Remove job-related data from the database
            try:
                backend_data["db_api"].drop_table(job_name)
                print(f"Successfully dropped database table for job '{job_name}'.")
            except Exception as e:
                print(f"Error dropping database table for job '{job_name}': {e}")
                traceback.print_exc()

            # Remove the job from the list of running jobs
            backend_data["running-jobs"].remove(job)
            print(f"Removed job '{job_name}' from the list of running jobs.")
            # --- End Job Cancellation Logic ---
            
            # --- Directory Deletion Logic ---
            # Construct the full path to the job's output directory
            job_output_directory = os.path.join(OUTPUT_DIR, job_name)
            print(f"Attempting to delete directory: {job_output_directory}")

            try:
                # Check if the directory exists
                if os.path.exists(job_output_directory) and os.path.isdir(job_output_directory):
                    # Recursively delete the directory and all its contents
                    shutil.rmtree(job_output_directory)
                    print(f"Successfully deleted directory: {job_output_directory}")
                elif not os.path.exists(job_output_directory):
                    print(f"Directory not found (already deleted or never existed): {job_output_directory}")
                else:
                    # This case handles if the path exists but is not a directory
                    print(f"Path exists but is not a directory: {job_output_directory}. Manual check required.")
            except Exception as e:
                print(f"Error deleting directory {job_output_directory}: {e}")
                traceback.print_exc()
            # --- End Directory Deletion Logic ---
            break # Exit loop once the job is found and processed
    if not job_found:
        print(f"Job '{job_name}' not found in running jobs.")

if __name__ == "__main__": 
    main()