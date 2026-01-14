# Base Image: Python 3.12 Slim (Optimized for AI)
FROM python:3.12-slim

# Install system dependencies (required for TensorFlow/OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python Dependencies
COPY requirements.txt .
# Increase timeout for heavy ML packages
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy ML Code
COPY . .

# Expose Port
EXPOSE 5000

# Run Flask App
CMD ["python", "app.py"]
