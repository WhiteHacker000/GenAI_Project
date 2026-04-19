#!/bin/bash

# Port for our backend
BACKEND_PORT=8000

# Start the FastAPI backend in the background
echo "🚀 Starting FastAPI backend on port $BACKEND_PORT..."
python3 -m uvicorn api:app --host 0.0.0.0 --port $BACKEND_PORT &

# Save the backend PID
BACKEND_PID=$!

# Logic to wait for the backend to be healthy
echo "⏳ Waiting for backend to initialize (loading model and FAISS)..."
MAX_RETRIES=30
COUNT=0
UP=0

while [ $COUNT -lt $MAX_RETRIES ]; do
  if curl -s http://127.0.0.1:$BACKEND_PORT/ > /dev/null; then
    echo "✅ Backend is UP and Healthy! Starting Streamlit..."
    UP=1
    break
  fi
  
  # Check if backend process is still running
  if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend process crashed on startup! Checking logs..."
    wait $BACKEND_PID
    exit 1
  fi

  echo "  (Attempt $((COUNT+1))/$MAX_RETRIES: Still waiting...)"
  sleep 2
  COUNT=$((COUNT+1))
done

if [ $UP -eq 0 ]; then
  echo "⚠️ Backend failed to start within 60 seconds. Proceeding anyway, but errors may occur."
fi

# Start the Streamlit frontend in the foreground
# Render uses port 10000 for public traffic by default
echo "🎨 Starting Streamlit frontend on port 10000..."
streamlit run app.py --server.port 10000 --server.address 0.0.0.0
