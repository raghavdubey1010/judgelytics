# Judgelytics

An AI-powered legal analytics and assistance platform. This repository contains both the backend API (FastAPI) and the frontend interface (HTML/JS/CSS).

## Quick Start for Mac/Linux Users

We've provided a simple bash script that handles all setup, creates a virtual environment, installs dependencies, and runs both the backend and frontend servers with a single command.

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd judgelytics
   ```

2. **Run the application:**
   ```bash
   bash run.sh
   ```

3. **Access the application:**
   - The **Frontend** will be available at `http://127.0.0.1:3000`
   - The **Backend API** will be running at `http://127.0.0.1:8000`

To stop both servers, simply press `Ctrl+C` in your terminal.

## Requirements
- Python 3.9+
- The `.env` file is included so the application works out of the box.

## Directory Structure
- `backend/` - FastAPI backend application and endpoints.
- `frontend/` - Static HTML/JS/CSS frontend.
- `ml_pipeline/` - Machine learning scripts and saved models.
- `run.sh` - Simple start script for Mac/Linux environments.
