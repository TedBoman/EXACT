from typing import List, Optional
import numpy as np
from db_interface import DBInterface
import pandas as pd
from datetime import datetime, timezone
import psycopg2
import psycopg2.extras as extras
from psycopg2.extras import execute_values
import multiprocessing as mp
from time import sleep

class TimescaleDBAPI(DBInterface):
    # Initialize the connection string that psycopg2 uses to connect to the database
    def __init__(self, conn_params: dict):
        user = conn_params["user"]
        password = conn_params["password"]
        host = conn_params["host"]
        port = conn_params["port"]
        database = conn_params["database"]
    
        self.connection_string = f'postgres://{user}:{password}@{host}:{port}/{database}'
    
    # Creates a hypertable called table_name with column-names columns copied from dataset
    # Also adds columns is_anomaly and injected_anomaly
    def create_table(self, table_name: str, columns: List[str]) -> Optional[str]:
        """
        Creates a new table with an auto-incrementing ID, a timestamp column,
        user-defined numeric columns, and standard boolean flag columns.
        The primary key will be a composite of the timestamp column and the ID.
        Converts the table to a TimescaleDB hypertable.

        Args:
            table_name: The name of the table to create.
            columns: A list of column names. The first column in this list
                     is treated as the primary timestamp column for hypertable
                     partitioning AND part of the composite primary key.
                     The rest are treated as NUMERIC data columns.

        Returns:
            The table_name if creation was successful, None otherwise.
        """
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            # Check if table exists
            check_exists_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                );
            """
            cursor.execute(check_exists_query, (table_name,))
            table_exists = cursor.fetchone()[0]

            if table_exists:
                print(f"Info: Table '{table_name}' already exists.")
                return None

            if not columns:
                print(f"Error: No columns provided for table '{table_name}'. A timestamp column is required.")
                return None

            # --- Define column structures ---

            # 1. Auto-incrementing ID column (will be part of the composite PK)
            # Note: PRIMARY KEY is removed from here; it will be defined as a composite key later.
            id_column_definition = '"id" BIGINT GENERATED ALWAYS AS IDENTITY'
            
            final_column_definitions = [id_column_definition]

            # 2. Timestamp column (from the first item in the input 'columns' list)
            # This column will be used for hypertable partitioning and be part of the PK.
            timestamp_col_name_from_input = columns[0]
            final_column_definitions.append(f'"{timestamp_col_name_from_input}" TIMESTAMPTZ NOT NULL')

            # 3. Other user-defined data columns (as NUMERIC)
            for i in range(1, len(columns)):
                final_column_definitions.append(f'"{columns[i]}" NUMERIC')
            
            # 4. Standard boolean flag columns
            final_column_definitions.append("is_anomaly BOOLEAN DEFAULT FALSE")
            final_column_definitions.append("injected_anomaly BOOLEAN DEFAULT FALSE")

            # 5. Define the Composite Primary Key
            # This includes the timestamp column and the auto-incrementing id.
            # The order (timestamp, id) is generally good for time-series queries.
            # Ensure the names used here exactly match how they are defined (including quotes if needed).
            composite_pk_definition = f'PRIMARY KEY ("{timestamp_col_name_from_input}", "id")'
            final_column_definitions.append(composite_pk_definition)
            
            # --- Create Table ---
            query_create_table = f'CREATE TABLE "{table_name}" ({", ".join(final_column_definitions)});'
            print(f"Executing: {query_create_table}")
            cursor.execute(query_create_table)
            
            # --- Make it a Hypertable ---
            # Use the raw name of the timestamp column for create_hypertable
            hypertable_partition_column = timestamp_col_name_from_input
            
            query_create_hypertable = f"SELECT create_hypertable('{table_name}', '{hypertable_partition_column}');"
            print(f"Executing: {query_create_hypertable}")
            cursor.execute(query_create_hypertable)

            conn.commit()
            print(f"Table '{table_name}' created successfully with composite PK, ID, and converted to hypertable.")
            return table_name

        except psycopg2.Error as db_err:
            print(f"Database error during table creation for '{table_name}': {db_err}")
            if conn:
                conn.rollback()
            return None
        except Exception as e:
            print(f"An unexpected error occurred during table creation for '{table_name}': {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()

    def insert_data(self, table_name: str, data: pd.DataFrame):
        """
        Inserts data into the specified table and sets the "injected_anomaly" column.

        Args:
            table_name: The name of the table.
            data: The DataFrame containing the data to insert.
            isAnomaly: A boolean indicating whether the data has been injected with an anomaly.
        """
        conn = psycopg2.connect(self.connection_string) # Connect to the database
        cursor = conn.cursor()
        with conn.cursor() as cur:
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
                conn.commit()
            except Exception as e:
                conn.rollback()
    
    # Reads each row of data in the table table_name that has a timestamp greater than or equal to time
    def read_data(self, from_time: datetime, table_name: str, to_time: datetime=None) -> pd.DataFrame:
        # Assuming the docker container is started, connect to the database
        try:
            conn = psycopg2.connect(self.connection_string)

            params = {}
            from_dt_utc_naive = from_time.astimezone(timezone.utc).replace(tzinfo=None)
            params['from_ts'] = from_dt_utc_naive

            if to_time is not None:
                to_dt_utc_naive = to_time.astimezone(timezone.utc).replace(tzinfo=None)
                params['to_ts'] = to_dt_utc_naive

            if to_time is not None:
                query = f'SELECT * FROM {table_name} WHERE timestamp >= \'{from_time}\' AND timestamp <= \'{to_time}\' ORDER BY timestamp ASC;'
            else:
                query = f'SELECT * FROM {table_name} WHERE timestamp >= \'{from_time}\' ORDER BY timestamp ASC;'

            df = pd.read_sql_query(query, conn, params=params) # Let pandas handle it

            print(f"Read data with columns: {df.columns.values}")

            return df
        except Exception as error:
            print("Error: %s" % error)
            conn.close()
            

    # Deletes the table_name table along with all its data
    def drop_table(self, table_name: str) -> None:
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            cursor.execute(f'DROP TABLE {table_name};')
            conn.commit()
        except Exception as error:
            print("Error: %s" % error)
            conn.close()
        finally:
            conn.close()

    # Checks if the table_name table exists in the database
    def table_exists(self, table_name: str) -> bool:
        result = []
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '{table_name}'"

            cursor.execute(query)
            result = cursor.fetchall()
        except Exception as error:
            print("Error: %s" % error)
            #conn.close()
        finally:
            #conn.close()
            if len(result) > 0:
                return True
            else:
                return False

    # Returns a list of all columns in the table_name table
    def get_columns(self, table_name: str) -> list[str]:
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"

            cursor.execute(query)
            result = cursor.fetchall()

        except Exception as error:
            print("Error: %s" % error)
            conn.close()
        finally:
            conn.close()
            columns = [x[0] for x in result]
            columns.remove("is_anomaly")
            columns.remove("injected_anomaly")

            return columns

    # Updates rows of the table that have an anomaly detected
    def update_anomalies(self, table_name: str, pk_column_name, anomaly_pk_values) -> None:
        updated_row_count_sum = 0
        try: 
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            for pk_value in anomaly_pk_values:
                # Construct query for a single row update
                # Table and column names are quoted for safety (e.g. if they have spaces or are keywords)
                # The actual pk_value is passed as a parameter to prevent SQL injection.
                query = f"UPDATE \"{table_name}\" SET is_anomaly = TRUE WHERE \"{pk_column_name}\" = %s;"
                
                
                try:
                    cursor.execute(query, (pk_value,)) # Parameters must be a tuple
                    updated_row_count_sum += cursor.rowcount
                except psycopg2.Error as single_exec_err:
                    print(f"Database error during single update for PK {pk_value} in table '{table_name}': {single_exec_err}")
                    # Decide on error strategy: continue with others, or rollback and raise?
                    # For now, we'll let it try others, and the final commit/rollback handles the transaction.
                    # If one fails, the whole transaction will be rolled back in the outer except block.

            conn.commit()
            # print(f"Attempted to update {len(anomaly_pk_values)} anomaly records in table '{table_name}'.")
            # print(f"Total rows reported as affected by database (sum of individual updates): {updated_row_count_sum}.")

        except psycopg2.Error as db_err: # Catch specific psycopg2 errors
            print(f"Database error during anomaly update for table '{table_name}': {db_err}")
            if conn:
                conn.rollback()
        except Exception as e:
            print(f"An unexpected error occurred during anomaly update for table '{table_name}': {e}")
            if conn: # Check if connection was established before trying to rollback
                conn.rollback()
        finally:
            if conn:
                conn.close()
            
    def list_all_tables(self):
        tables = []
        query = """
            SELECT tablename
            FROM pg_catalog.pg_tables
            WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema'
              AND (tablename LIKE 'job_batch_%' OR tablename LIKE 'job_stream_%');
        """
        # Or query without the LIKE filter and filter in Python if preferred
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            tables = [row[0] for row in results]
            conn.close()
        except Exception as e:
            print(f"Error listing tables: {e}")
            # Handle error appropriately, maybe reconnect?
        return tables