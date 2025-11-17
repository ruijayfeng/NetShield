#!/bin/bash
# NetShield startup script

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/app/src"
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false

# Create directories if they don't exist
mkdir -p data/raw data/processed data/models output logs

# Start the application
echo "Starting NetShield on port ${PORT:-8501}..."
streamlit run src/visualization/dashboard.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false