import pandas as pd
import psycopg2
import os
from datetime import datetime, timezone
import time
from dotenv import load_dotenv
from timescaledb_api import TimescaleDBAPI

load_dotenv()
HOST = 'localhost'
PORT = int(os.getenv('DATABASE_PORT'))
USER = os.getenv('DATABASE_USER')
PASSWORD = os.getenv('DATABASE_PASSWORD')
NAME = os.getenv('DATABASE_NAME')
BACKEND_HOST = os.getenv('BACKEND_HOST')
BACKEND_PORT = int(os.getenv('BACKEND_PORT'))

# Assuming the docker container is started, connect to the database
conn_params = {
    "user": USER,
    "password": PASSWORD,
    "host": HOST,
    "port": PORT,
    "database": NAME
}

db_api = TimescaleDBAPI(conn_params)

df = db_api.read_data(datetime.fromtimestamp(200), "test")    # Read the data from the database
print(df)
df["is_anomaly"] = False