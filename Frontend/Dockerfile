FROM python:3.10-slim

ARG FRONTEND_PORT

WORKDIR /app

COPY ./Frontend/ /app/

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY ./Backend/api/api.py /app/
COPY ./Docker/.env /app/
RUN mkdir -p /app/graphs

EXPOSE ${FRONTEND_PORT}

RUN chmod +x entrypoint.sh

CMD ["bash", "entrypoint.sh"]
