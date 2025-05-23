# syntax = docker/dockerfile:1
FROM python:3.10-slim

ARG BACKEND_PORT

RUN apt-get update && \
    apt-get install -y build-essential g++ && \
    # Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*

COPY ./Backend/requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

EXPOSE ${BACKEND_PORT}

COPY ./Backend/ .
COPY ./Database/db_interface.py .
COPY ./Database/timescaledb_api.py .
COPY ./Docker/.env .
COPY ./Docker/.env ./api/

ENTRYPOINT ["python", "-u", "engine.py"]
