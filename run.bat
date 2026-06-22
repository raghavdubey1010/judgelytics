@echo off
echo Starting Judgelytics setup and run script for Windows...

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing backend dependencies...
pip install -r backend\requirements.txt

if not exist ".env" (
    echo Creating .env from .env.example...
    copy .env.example .env
)

if not exist "ml_pipeline\saved_models" mkdir "ml_pipeline\saved_models"
if not exist "backend\app\data" mkdir "backend\app\data"

echo Starting FastAPI backend...
start "Judgelytics Backend" cmd /c "cd backend && title Judgelytics Backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo Starting Frontend server...
start "Judgelytics Frontend" cmd /c "cd frontend && title Judgelytics Frontend && python -m http.server 3000"

echo Setup complete! Judgelytics is running in separate command windows.
echo Frontend available at: http://127.0.0.1:3000
echo Backend API available at: http://127.0.0.1:8000
