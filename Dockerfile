FROM python:3.11-slim-buster

COPY requirements/ /app/requirements/

RUN pip install --no-cache-dir -r /app/requirements/prod.txt

COPY api /app/api

WORKDIR /app

EXPOSE 8001