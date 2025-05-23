from typing import List, Dict, Union

class AnomalySetting:
    def __init__(self, anomaly_type: str, timestamp: int, magnitude: int, 
                 percentage: int, columns: List[str] = None, duration: str = None,
                 data_range: List[float] = None, mean: List[float] = None):
        self.anomaly_type = anomaly_type
        self.timestamp = timestamp
        self.magnitude = magnitude
        self.percentage = percentage
        self.duration = duration
        self.columns = columns
        self.data_range = data_range
        self.mean = mean

class Job:
    def __init__(self, filepath: str, simulation_type,speedup: int = None, anomaly_settings: List[AnomalySetting] = None, table_name: str = None, debug=False):
        self.filepath = filepath
        self.anomaly_settings = anomaly_settings
        self.simulation_type = simulation_type
        self.speedup = speedup
        self.table_name = table_name
        self.debug = debug