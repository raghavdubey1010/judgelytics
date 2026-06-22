#!/bin/bash

echo "🚀 Starting Judgelytics setup and run script for Mac/Linux..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 could not be found. Please install Python 3."
    exit 1
fi

# Setup environment variables
if [ ! -f ".env" ]; then
    echo "📄 Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your GROQ_API_KEY before using the chat/legal features."
fi

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate the virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "📥 Installing backend dependencies..."
pip install -r backend/requirements.txt

# Create necessary directories
mkdir -p ml_pipeline/saved_models
mkdir -p backend/app/data

# Start the FastAPI backend in the background
echo "⚙️  Starting FastAPI backend..."
cd backend
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Start a simple HTTP server for the frontend
echo "🌐 Starting Frontend server on http://127.0.0.1:3000..."
cd frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!
cd ..

echo "✅ Setup complete! Judgelytics is running."
echo "👉 Frontend available at: http://127.0.0.1:3000"
echo "👉 Backend API available at: http://127.0.0.1:8000"
echo "Press Ctrl+C to stop both servers."

# Trap Ctrl+C to kill both background processes
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT SIGTERM

# Wait indefinitely
wait
