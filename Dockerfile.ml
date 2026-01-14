# Base Image: Python 3.12
FROM python:3.12-slim

# Install System Deps (for TensorFlow/Pillow)
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install ML Dependencies
COPY ml_service/requirements.txt ./ml_service/requirements.txt
RUN pip install --no-cache-dir -r ml_service/requirements.txt

# Copy ML Code
COPY ml_service ./ml_service
# Copy Model (If generating dynamically, user must train or commit weights)
# We assume weights are in ml_service/ or generated on fly
COPY ml_service/model.py ./
COPY ml_service/app.py ./microservice_app.py

# Env
ENV PORT=5000

# Start
# We run app.py directly (ensure it uses host='0.0.0.0')
CMD ["python", "ml_service/app.py"]
