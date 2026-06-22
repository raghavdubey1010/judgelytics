# Bulletproof Mac Setup Guide for Judgelytics

> **ATTENTION MAC COPILOT/USER**: If the application is failing to start, it is almost certainly due to a corrupted python virtual environment, a missing `.env` file, or incomplete dependency installation.
> 
> **DO NOT skip any of these steps.** Run these exact commands in order to guarantee a successful boot.

## Step 1: Wipe the Slate Clean
We must delete the existing virtual environment to ensure no corrupted dependencies are causing issues.

Run this command in the project root:
```bash
rm -rf .venv
```

## Step 2: Create a Fresh Virtual Environment
Macs use `python3`. Recreate the virtual environment and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> **CAUTION**: Ensure you see `(.venv)` in your terminal prompt before proceeding! If you do not see it, the environment is not active and the next steps will fail.

## Step 3: Install Dependencies
Install all required backend dependencies. We are forcefully updating `pip` and installing `anthropic` explicitly to ensure the MCP tool features work.

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install anthropic
```

## Step 4: Configure Environment Variables
The `.env` file is untracked by Git, so the Mac does not have it by default. 

1. Create a new `.env` file in the root directory.
2. Open it and paste the following EXACT structure:
```env
# Database Settings
DATABASE_URL=sqlite+aiosqlite:///./judgelytics.db

# Security Settings
SECRET_KEY=your_super_secret_key_change_in_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# AI Model API Keys
ANTHROPIC_API_KEY=your_anthropic_key_here
GROQ_API_KEY=your_groq_key_here
```
3. **CRITICAL:** Ask the user to provide their real `ANTHROPIC_API_KEY` (and `GROQ_API_KEY` if available) and update the file.

## Step 5: Start the Backend Server
Keep your terminal open with the `(.venv)` activated. Start the backend:

```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
> **TIP**: Wait until you see `Application startup complete.` in the terminal logs. 
> To verify, open your browser and navigate to `http://127.0.0.1:8000/api/v1/health`. You should see `{"status":"ok"}`.

## Step 6: Start the Frontend Server
Open a **new, separate terminal window**. You do *not* need the virtual environment for this step.

Navigate to the project root and start a simple HTTP server for the frontend:
```bash
cd frontend
python3 -m http.server 3000
```

## Step 7: Access the Application
Open Google Chrome or Safari and go to:
`http://localhost:3000`

If you face any `401 Unauthorized` errors when clicking buttons, it means there is an old JWT token in the browser. Instruct the user to **Log Out** and **Log In** again to generate a fresh session.
