#!/bin/bash

# --------------------------------------------------
# Script to Restart the Flask Server on Raspberry Pi
# --------------------------------------------------

# Exit immediately if a command exits with a non-zero status
set -e

# Define the project directory
PROJECT_DIR="/home/tranductri2003/Code/PBL06_multi-speaker-voice-cloning/PBL06_multi-speaker-voice-cloning-backend"

# Define the virtual environment directory
VENV_DIR="$PROJECT_DIR/venv"

# Define the Flask application file
FLASK_APP_FILE="app.py"

# Define the command to run the Flask application
FLASK_CMD="python3 $FLASK_APP_FILE"

# Define the log file for the Flask application
LOG_FILE="$PROJECT_DIR/flask_app.log"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Project directory not found! Exiting."; exit 1; }

# Activate the virtual environment if it exists; otherwise, create it
if [ -d "$VENV_DIR" ]; then
    echo "Activating existing virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment not found. Creating a new one..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install or update dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Stop any running instances of the Flask application
echo "Stopping existing Flask application instances..."
pkill -f "$FLASK_CMD" || echo "No existing Flask instances running."

# Start the Flask application in the background and redirect output to the log file
echo "Starting Flask application..."
nohup $FLASK_CMD > "$LOG_FILE" 2>&1 &

# Deactivate the virtual environment
deactivate

echo "Flask application has been restarted successfully."
