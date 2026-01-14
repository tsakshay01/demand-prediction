#!/bin/bash

# Start ML Service in Background
echo "🚀 Starting AI Brain (Python)..."
python ml_service/app.py &

# Wait a few seconds for ML to warm up
sleep 5

# Start Web Server in Foreground
echo "🌐 Starting Web Server (Node.js)..."
node server.js
