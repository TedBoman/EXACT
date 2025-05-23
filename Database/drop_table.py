import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
HOST = 'localhost'
PORT = int(os.getenv('DATABASE_PORT'))
USER = os.getenv('DATABASE_USER')
PASSWORD = os.getenv('DATABASE_PASSWORD')
NAME = os.getenv('DATABASE_NAME')

connection_string = f'postgres://{USER}:{PASSWORD}@{HOST}:{PORT}/{NAME}'
conn = psycopg2.connect(connection_string)
cursor = conn.cursor()

query = f'DROP TABLE system;'
cursor.execute(query)
conn.commit()
conn.close()