# Backend API and CLI-tool documentation

Found in the "./Backend/api/" directory, we have a API to interact with the backend that is used by our Fronted as well as our CLI-tool.

## CLI-tool

We've made it possible to send requests to the backend with the backend api through the command-line. This is made possible by the invoking "./Backend/api/cli_tool.py". The main function parses the command-line arguments and calls, handles errors and calls appropiate function sending data to the backend. Run "python cli_tool.py help" for CLI-tool documentation.

The options used for the CLI-tool are:
- run-batch
- run-stream
- get-data
- get-running
- cancel
- get-models
- get-injection-methods
- get-datasets
- get-all-jobs
- import-datasets

Most of the options will ask the user to provide details needed for that specific option. E.g., the "run-batch" option requires the user to specify which model to use for detection, on what dataset they want to do detection on, what to name the batch job and prompt for anomaly injection details if they choose to inject anomalies.

## BackendAPI

To use the backend api, first you'll need to import the BackendAPI class and then instantiate a BackendAPI object providing it with a host and port

```py
from api import BackendAPI

HOST = "secret"
PORT = 1234

api = BackendAPI(HOST, PORT)
```

Class designed to serve all requests/responses to and from the backend. It is initialized with a host and port sent to the constructor to use for connecting to the backend socket. 

Each method defined in the class takes a set of input parameters defined for that specific request. A python dictionary is then formatted, converted into a json string and then sent to the backend through the defined "__send_data" method.

### Example method, run_batch

The run_batch method defined:

```py
# Sends a request to the backend to start a batch job
def run_batch(self, model: str, dataset: str, name: str, inj_params: dict=None) -> None:
    data = {
        "METHOD": "run-batch",
        "model": model,
        "dataset": dataset,
        "name": name
    }
    if inj_params:
        data["inj_params"] = inj_params

    self.__send_data(data, response=False)
```
Where "inj_params" is a dictionary defined as: 
```py
inj_params = {
    "anomaly_type": anomaly_type,
    "timestamp": timestamp,
    "magnitude": magnitude,
    "percentage": percentage,
    "duration": duration,
    "columns": columns
}
```

### __send_data method

The __send_data function initiates a socket object, connects to the backend engine and then sends the json string through the socket. The client will wait for a response from the backend unless the response parameter is set to False. If method is "import-dataset" two messages is sent, one contains the request dictionary and one contains the file contents of the dataset. 

```py
# Initates connection to backend and sends json data through the socket
def __send_data(self, data: str, response: bool=True) -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))

        # Send two messages through the same connection if the method is import-dataset
        if data["METHOD"] == "import-dataset":
            file_content = data["file_content"]
            del data["file_content"]
            data = json.dumps(data)
            sock.sendall(bytes(data, encoding="utf-8"))
            sleep(0.5)
            sock.sendall(bytes(file_content, encoding="utf-8"))
        elif data["METHOD"] == "get-data":       
            data = json.dumps(data)
            sock.sendall(bytes(data, encoding="utf-8"))

            recv_data = sock.recv(1024).decode("utf-8")
            json_data = recv_data
            while recv_data:
                sock.settimeout(1)
                recv_data = sock.recv(1024).decode("utf-8")
                if recv_data:
                    json_data += recv_data
            
            data = json.loads(json_data)
            return data
        else:
            data = json.dumps(data)
            sock.sendall(bytes(data, encoding="utf-8"))
        if response:
            data = sock.recv(1024)
            data = data.decode("utf-8")
            return data
    except Exception as e:
        print(e)
```

### All method declarations for BackendAPI class

For the purposes of defining what the BackendAPI provides, here is a list of all instance methods. These are created to facilitate all the interaction needed from a users perspective and is used by the Fronted as well as able to be used from the CLI-tool. 

```py
def __init__(self, host: str, port: int) -> None:
def run_batch(self, model: str, dataset: str, name: str, inj_params: dict=None) -> None:
def run_stream(self, model: str,dataset: str, name: str, speedup: int, inj_params: dict=None) -> None:
def get_data(self, timestamp: str, name: str) -> str:
def get_running(self) -> str:
def cancel_job(self, name: str) -> None:
def get_models(self) -> str:
def get_injection_methods(self) -> str:
def get_datasets(self) -> str:
def get_all_jobs(self) -> str:
def import_dataset(self, file_path: str, timestamp_column: str) -> None:
```