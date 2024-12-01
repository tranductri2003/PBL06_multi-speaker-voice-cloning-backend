#!/bin/bash

# --------------------------------------------------
# Script to Restart the FastAPI Server on Raspberry Pi
# --------------------------------------------------

# Exit immediately if a command exits with a non-zero status
set -e

# Define the project directory
PROJECT_DIR="/home/tranductri2003/Code/PBL06_multi-speaker-voice-cloning/PBL06_multi-speaker-voice-cloning-backend"

# Define the virtual environment directory
VENV_DIR="$PROJECT_DIR/venv"

# Define the FastAPI application module
FASTAPI_APP_MODULE="app:app"

cd "$PROJECT_DIR" || { echo "Project directory not found! Exiting."; exit 1; }

cd src

# Define the command to run the FastAPI application using Uvicorn
UVICORN_CMD="uvicorn $FASTAPI_APP_MODULE --host 0.0.0.0 --port 8000"

# Define the log file for the FastAPI application
LOG_FILE="$PROJECT_DIR/fastapi_app.log"

# Activate the virtual environment if it exists; otherwise, create it
if [ -d "$VENV_DIR" ]; then
    echo "Activating existing virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment not found. Creating a new one..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

# Stop any running instances of the FastAPI application
echo "Stopping existing FastAPI application instances..."
pkill -f "$UVICORN_CMD" || echo "No existing FastAPI instances running."

# Start the FastAPI application in the background and redirect output to the log file
echo "Starting FastAPI application..."
nohup $UVICORN_CMD > "$LOG_FILE" 2>&1 &

deactivate

echo "FastAPI application has been restarted successfully."
