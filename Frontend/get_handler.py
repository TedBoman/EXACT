from frontend_handler import FrontendHandler
from dotenv import load_dotenv
import os

load_dotenv()
HOST = 'Backend'
PORT = int(os.getenv('BACKEND_PORT'))

def get_handler():
    # Create instance of FrontendHandler

    handler = FrontendHandler(HOST, PORT)
    return handler