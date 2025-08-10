FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && apt-get install -y --no-install-recommends build-essential libhdf5-dev libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools/wheel to latest
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY . .

# Запуск uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
