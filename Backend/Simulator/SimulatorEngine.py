from pathlib import Path
import sys
import pandas as pd
import os

# Custom
from Simulator.SimulateFromDataSet.simulator import Simulator
from Simulator.BatchImport.batchimport import BatchImporter
from Simulator.DBAPI import type_classes as tc
from Simulator.DBAPI.debug_utils import DebugLogger as dl

DEFAULT_PATH = './Datasets/'

class SimulatorEngine:
    def process_file(self, file_path, conn_params, simulation_type, anomaly_settings, start_time, speedup: int = 1, table_name = None, timestamp_col_name=None, label_col_name=None):
        """Processes a single file based on the specified use case."""
        dl.debug_print("Starting to process file!")
        match simulation_type:
            case "stream":
                print("Starting stream job...")
                sys.stdout.flush()
                try:
                    file_extension = Path(file_path).suffix
                    sim = Simulator(file_path, file_extension, start_time, x_speedup=speedup)
                    return sim.start_simulation(conn_params=conn_params, anomaly_settings=anomaly_settings, table_name=table_name, timestamp_col_name=timestamp_col_name, label_col_name=label_col_name)
                except Exception as e:
                    dl.print_exception(f"Error: {e}")
                    return 0
            case "batch":
                print("Starting batch job...")
                sys.stdout.flush()
                try:
                    file_extension = Path(file_path).suffix
                    sim = BatchImporter(file_path, file_extension, start_time, 5)
                    return sim.start_simulation(conn_params=conn_params, anomaly_settings=anomaly_settings, table_name=table_name, timestamp_col_name=timestamp_col_name, label_col_name=label_col_name)
                except Exception as e:
                    dl.print_exception(f"Error: {e}")
                    return 0

    def main(self, db_conn_params, job, timestamp_col_name=None, label_col_name=None):
        # Set debug mode once for all files
        dl.set_debug(False)  # or False to disable debug prints

        # Check if the path exists
        if os.path.isfile(job.filepath):
            # Filepath is valid, do nothing
            pass
        else:
            # Prepend the default path to the filename
            job.filepath = os.path.join(DEFAULT_PATH, os.path.basename(job.filepath))

        if job:
            # Convert anomaly settings timestamps to datetime objects
            if isinstance(job, (tc.Job)) and job.anomaly_settings:
                return self.process_file(job.filepath, db_conn_params, job.simulation_type, job.anomaly_settings, pd.to_timedelta(0), job.speedup, job.table_name if job.table_name else None, timestamp_col_name, label_col_name)
            else:
                return self.process_file(job.filepath, db_conn_params, job.simulation_type, job.anomaly_settings, pd.to_timedelta(0), job.speedup, job.table_name if job.table_name else None, timestamp_col_name, label_col_name)