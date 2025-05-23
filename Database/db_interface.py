from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from datetime import datetime

class DBInterface(ABC):
    # Constructor that adds all connection parameters needed to connect to the database to the object
    # conn_params is a dictionary with the parameters needed for the specific database implementation
    # As an example structure, the dictionary for a TimescaleDB implementation of the DBInterface could look like this:
    # {
    #     "user": "username",
    #     "password": "password",
    #     "host": "hostname",
    #     "port": "port",
    #     "database": "database"
    # }
    @abstractmethod
    def __init__(self, conn_params: dict):
        pass
    # Creates a hypertable called table_name with column-names columns
    # First column of name columns[0] is of type TIMESTAMPTZ NOT NULL and the rest are VARCHAR(50)
    # Then two new columns of type BOOLEAN are added to the table, is_anomaly and injected_anomaly
    @abstractmethod
    def create_table(self, table_name: str, columns: List[str]) -> Optional[str]:
        pass
    # Inserts data into the table_name table. The data is a pandas DataFrame with matching columns to the table
    @abstractmethod
    def insert_data(self, table_name: str, data: pd.DataFrame) -> None:
        pass
    # Reads each row of data in the table table_name that has a timestamp greater than or equal to time
    @abstractmethod
    def read_data(self, from_time: datetime, table_name: str, to_time: datetime=None) -> pd.DataFrame:
        pass
    # Deletes the table_name table along with all its data
    @abstractmethod
    def drop_table(self, table_name: str) -> None:
        pass
    # Checks if the table_name table exists in the database
    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        pass
    # Returns a list of all columns in the table_name table
    @abstractmethod
    def get_columns(self, table_name: str) -> list[str]:
        pass
    # Updates rows of the table that have an anomaly detected
    @abstractmethod
    def update_anomalies(self, table_name: str, pk_column_name, anomaly_pk_values) -> None:
        pass
    # Lists all tables in the database
    @abstractmethod
    def list_all_tables(self) -> None:
        pass