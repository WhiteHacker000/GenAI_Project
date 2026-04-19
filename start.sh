#!/bin/bash

# Start the FastAPI backend in the background on port 8000
echo "Starting FastAPI backend..."
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds for the backend to initialize
sleep 5

# Start the Streamlit frontend in the foreground on port 10000
# We use 10000 because it is Render's default port for HTTP traffic
echo "Starting Streamlit frontend..."
streamlit run app.py --server.port 10000 --server.address 0.0.0.0
