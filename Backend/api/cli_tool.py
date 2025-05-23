# This CLI tool is a remaining feature of EXACT before XAI was implemented.
# The tool may or may not work will the new XAI integration due to the
# magnitudes of the changed made to the backend.

# import sys
# import os
# from datetime import datetime, timezone
# from api import BackendAPI
# import json
# import pandas as pd
# from io import StringIO

# from run_batch import run_batch
# from run_stream import run_stream
# from dotenv import load_dotenv

# dotenv_path = "../../Docker/.env"

# load_dotenv(dotenv_path)
# HOST = 'localhost'
# PORT = int(os.getenv('BACKEND_PORT'))

# DOC = """"python cli-tool.py run-batch"
# starts anomaly detection of batch data after user has been prompted to enter details of the job

# "python cli-tool.py run-stream" 
# starts anomaly detection of stream data after user has been prompted to enter details of the job

# "python cli-tool.py get-data <from_timestamp> <to_timestamp> <name>"
# get processed data from <name>, meaning just the data that has gone through our detection model. <from_timestamp> allows for filtering of data. All timestamps are in seconds from epoch. If <to_timestamp> is not provided, all data from <from_timestamp> to now is returned.

# "python cli-tool.py get-running"
# get all running datasets

# "python cli-tool.py cancel-job <name>" 
# cancels the currently running batch or stream named <name>

# "python cli-tool.py get-models"
# gets all available models for anomaly detection

# "python cli-tool.py get-injection-methods"
# gets all available injection methods for anomaly detection

# "python cli-tool.py get-datasets"
# gets all available datasets

# "python cli-tool.py get-all-jobs"
# gets all started and/or running jobs

# "python cli-tool.py get-columns <name>"

# "python cli-tool.py import-dataset <dataset-file-path> <timestamp-column-name>"
# uploads a dataset to the backend by adding the file to the Dataset directory
        
# "python cli-tool.py help"
# prints this help message
# """

# # Main function handles argument parsing when the API is invoked from the command line
# def main(argv: list[str]) -> None:
#     result = None
#     arg_len = len(argv)
#     api = BackendAPI(HOST, PORT)
#     match argv[1]:
#         # Start a batch job in the backend if the command is "run-batch"
#         case "run-batch":
#             if arg_len != 2:
#                 handle_error(1, "Invalid number of arguments")

#             # Makes user input and sends request to the backend
#             run_batch(api)
            
#         # Start a stream job in the backend if the command is "run-stream"
#         case "run-stream":
#             if arg_len != 2:
#                 handle_error(1, "Invalid number of arguments")
            
#             # Makes user input and sends request to the backend
#             run_stream(api)
        
#         # Get data from a running job if the command is "get-data", the backend will return data that has gone through the detection model
#         case "get-data":
#             if (arg_len == 4):
#                 from_timestamp = argv[2]
#                 result = api.get_data(from_timestamp, argv[3])
#             elif (arg_len == 5):
#                 from_timestamp = argv[2]
#                 to_timestamp = argv[3]
#                 result = api.get_data(from_timestamp, argv[4], to_timestamp)
#             else:
#                 handle_error(1, "Invalid number of arguments")
        
#         # Inject anomalies into a running job if the command is "inject-anomaly"
#         case "inject-anomaly":
#             if (arg_len != 4):
#                 handle_error(1, "Invalid number of arguments")
#             timestamps = argv[2].split(',')
#             result = api.inject_anomaly(timestamps, argv[3])

#         # Print all running datasets if the command is "get-running"
#         case "get-running":
#             if (arg_len != 2):
#                 handle_error(1, "Invalid number of arguments")
#             result = api.get_running()

#         # Cancel a running job if the command is "cancel"
#         case "cancel-job":
#             if (arg_len != 3):
#                 handle_error(1, "Invalid number of arguments")
#             result = api.cancel_job(argv[2])

#         # Get all avaliable models for anomaly detection if the command is "get-models"
#         case "get-models":
#             if (arg_len != 2):
#                 handle_error(1, "Invalid number of arguments")
#             result = api.get_models()

#         # Get all avaliable models for anomaly detection if the command is "get-models"
#         case "get-xai-methods":
#             if (arg_len != 2):
#                 handle_error(1, "Invalid number of arguments")
#             result = api.get_xai_methods()

#         # Get all avaliable injection methods for anomaly detection if the command is "get-injection-methods"
#         case "get-injection-methods":
#             if (arg_len != 2):
#                 handle_error(1, "Invalid number of arguments")
#             result = api.get_injection_methods()

#         # Get all avaliable datasets if the command is "get-datasets"
#         case "get-datasets":
#             if (arg_len != 2):
#                 handle_error(1, "Invalid number of arguments")
#             result = api.get_datasets()
        
#         # Get all started and/or running jobs
#         case "get-all-jobs":
#             if (arg_len != 2):
#                 handle_error(1, "Invalid number of arguments")
#             result = api.get_all_jobs()

#         # Get columns of a running job
#         case "get-columns":
#             if (arg_len != 3):
#                 handle_error(1, "Invalid number of arguments")
#             result = api.get_columns(argv[2])

#         # Upload a dataset to the backend if the command is "import-dataset"
#         case "import-dataset":
#             if (arg_len != 4):
#                 handle_error(1, "Invalid number of arguments")
#             api.import_dataset(argv[2], argv[3])

#         # Print information about the backend API command line tool if the command is "help"
#         case "help":
#             print(DOC)

#         # Print an error message if the command is not recognized
#         case _: 
#             handle_error(3, f'argument "{argv[1]}" not recognized as a valid command')

#     # Print return messgage in terminal when API is used by the command line tool
#     if argv[1] == "get-data" and result:
#         df = pd.read_json(StringIO(result["data"]), orient="split")
#         print(df)
#     elif argv[1] != "help" and argv[1] and result:
#         print(f'Recieved from backend: {result}')
        
# def handle_error(code: int, message: str) -> None:
#         print(message)
#         exit(code) 

# if __name__ == "__main__":
#     main(sys.argv)