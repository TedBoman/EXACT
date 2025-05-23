# This run_batch file will not work with the current backend. It remains from the
# original EXACT tool without XAI integration.

# import json
# from api import BackendAPI

# def run_stream(api: BackendAPI) -> dict:
#     # Gather available models for detection
#     response = api.get_models()
#     models = json.loads(response)["models"]
#     print(f"Theses are the models provided by EXACT: {models}")
#     model = input("Enter the model to use: ")
#     while model not in models:
#         model = input("Model not found, please enter a valid model: ")

#     # Gather available datasets for detection
#     response = api.get_datasets()
#     datasets = json.loads(response)["datasets"]
#     print(f"Theses are the datasets provided by EXACT: {datasets}")
#     dataset = input("Enter the dataset to use: ")
#     while dataset not in datasets:
#         dataset = input("Dataset not found, please enter a valid dataset: ")

#     # Gather available jobs to make sure user gives a unique name
#     response = api.get_all_jobs()
#     jobs = json.loads(response)["jobs"]
#     name = input("Enter what you want to name the job: ")
#     if name in jobs:
#         name = input("Name already in use, please enter a new name: ")
        
#     # Gather speedup factor
#     speedup = int(input("Enter the speedup factor for the stream, as an integer value: "))

#     # Ask user if they want debug prints
#     debug = input("Enable debug prints (y/N): ")
#     if debug == "y" or debug == "Y":
#         debug = True
#     else:
#         debug = False

#     # Ask user if they want to inject an anomalies
#     insert_anomaly = input("Do you want to insert anomalies? (y/N): ")

#     inj_params = []  # Initialize as a list to store multiple anomalies
#     if insert_anomaly == "y":
#         # Gather injection methods for anomalies
#         response = api.get_injection_methods()
#         injection_methods = json.loads(response)["injection_methods"]
#         print(f"These are the injection methods provided by EXACT: {injection_methods}")

#         while True:  # Loop to allow input of multiple anomalies
#             injection_method = input("Enter injection method (or 'done' to finish): ")
#             if injection_method.lower() == 'done':
#                 break

#             while injection_method not in injection_methods:
#                 injection_method = input("Injection method not found, please enter a valid anomaly type: ")

#             timestamp = input("Enter the timestamp to start anomaly: ")
#             magnitude = input("Enter the magnitude of the anomaly: ")
#             duration = input("Enter a duration (e.g., '30s', '1H', '30min', '2D', '1h30m', '2days 5hours') or leave empty for a point anomaly: ")
#             percentage = input("Enter the percentage of data (during the duration, this percentage of points will be an anomaly): ")
#             columns_string = input("Enter the columns to inject anomalies into, as a comma separated list (a,b,c,d,...): ")

#             anomaly = {
#                 "anomaly_type": injection_method,
#                 "timestamp": timestamp,
#                 "magnitude": magnitude,
#                 "percentage": percentage,
#                 "duration": duration,
#                 "columns": columns_string.split(',')
#             }
#             inj_params.append(anomaly)  # Add the anomaly dictionary to the list

#         api.run_stream(model, dataset, name, speedup, debug, inj_params)  # Pass the list of anomalies
#     else:
#         api.run_stream(model, dataset, name, speedup, debug)