#!/bin/bash

# --------------------------------------------------
# Script to Restart the FastAPI Server on Raspberry Pi
# --------------------------------------------------

# Exit on error
set -e

# Project configuration
PROJECT_DIR="/home/tranductri2003/Code/PBL06_multi-speaker-voice-cloning/PBL06_multi-speaker-voice-cloning-backend"
VENV_DIR="$PROJECT_DIR/env"
SRC_DIR="$PROJECT_DIR/src"
SERVER_CMD="python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000"
LOG_FILE="$PROJECT_DIR/server.log"

echo "🚀 Starting deployment process..."

# Navigate to project directory
cd "$PROJECT_DIR" || { echo "❌ Project directory not found!"; exit 1; }

# Handle virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating new virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "✅ Using existing virtual environment..."
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Update dependencies
echo "📥 Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Navigate to source directory
echo "📂 Changing to source directory..."
cd "$SRC_DIR" || { echo "❌ Source directory not found!"; exit 1; }

# Stop existing server instances
echo "🛑 Stopping existing server instances..."
pkill -f "uvicorn app:app" || echo "ℹ️  No existing server running"

# Start server
echo "🌟 Starting server..."
nohup $SERVER_CMD > "$LOG_FILE" 2>&1 &

# Wait a bit to check if server started successfully
sleep 2

# Check if server is running
if pgrep -f "uvicorn app:app" > /dev/null; then
    echo "✨ Server started successfully!"
    echo "📝 Logs available at: $LOG_FILE"
else
    echo "❌ Failed to start server. Check logs at: $LOG_FILE"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo "🎉 Deployment completed!"
