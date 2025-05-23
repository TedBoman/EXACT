import psycopg2
from psycopg2.extras import execute_values
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
import numpy as np

class DBInterface(ABC):
    # Constructor that initiates a connection to the database and adds the connection to self.conn
    def __init__(self, conn_params: dict):
        """
        Initializes a connection to the TimescaleDB database.

        Args:
            conn_params (dict): A dictionary containing the connection parameters.
                                 Example: {"dbname": "your_db", "user": "your_user", ...}
        """

        try:
            CONNECTION = f"postgres://{conn_params['user']}:{conn_params['passwd']}@{conn_params.get('host', 'localhost')}:{conn_params.get('port', 5432)}/{conn_params['dbname']}"
            self.conn = psycopg2.connect(CONNECTION)
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"Could not connect to the database: {e}")
            return None

    # Creates a hypertable called table_name with column-names columns
    def create_table(self, table_name: str, columns: list[str]):
        """
        Creates a hypertable in the TimescaleDB database.

        Args:
            table_name (str): The name of the hypertable to create.
            columns (list[str]): A list of column names for the hypertable.
        """
        # Adjust types based on your actual data
        columns_with_types = columns.copy()
        first_column = f"{columns_with_types[0]} TIMESTAMP WITHOUT TIME ZONE"
        for i in range(len(columns_with_types)):
            columns_with_types[i] = f'"{columns_with_types[i]}"'
            # Assign the data type only once
            columns_with_types[i] = f'{columns_with_types[i]} DOUBLE PRECISION' 
        
        columns_with_types[0] = first_column

        # Add the "injected_anomaly" column
        columns_with_types.append('"injected_anomaly" BOOLEAN')

        query_create_table = f'CREATE TABLE "{table_name}" ({",".join(columns_with_types)});'
        self.cursor.execute(query_create_table)
        
        # Make the table a hypertable partitioned by timestamp
        query_create_hypertable = f'SELECT create_hypertable(\'{table_name}\', \'timestamp\');'
        self.cursor.execute(query_create_hypertable)

        self.conn.commit()

    def insert_data(self, table_name: str, data: pd.DataFrame):
        """
        Inserts data into the specified table and sets the "injected_anomaly" column.

        Args:
            table_name: The name of the table.
            data: The DataFrame containing the data to insert.
            isAnomaly: A boolean indicating whether the data has been injected with an anomaly.
        """
        with self.conn.cursor() as cur:
            # Add "injected_anomaly" to the columns
            columns = ', '.join([f'"{col}"' for col in data.columns])  
            query = f"INSERT INTO \"{table_name}\" ({columns}) VALUES %s"

            try:
                # Convert DataFrame to list of tuples, with type conversion and anomaly flag
                values = [tuple(
                    float(x) if isinstance(x, (np.float64, np.float32)) else x
                    for x in row
                ) for row in data.values]

                execute_values(cur, query, values)
                self.conn.commit()
                print(f"Data insertion successful for table: {table_name}")
            except Exception as e:
                self.conn.rollback()

    # Reads data from the table_name table
    def read_data(self, table_name: str, time: datetime):
        pass

    # Deletes the table_name table along with all its data
    def drop_table(self, table_name: str):
        pass