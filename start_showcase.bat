@echo off
echo =======================================================
echo    Nirikhshon Final Year Project Showcase Launcher
echo =======================================================
echo.

echo [1/2] Starting Flask Backend...
start cmd /k "cd backend && ..\tb_env\Scripts\python app.py"

echo [2/2] Starting Next.js Frontend...
start cmd /k "cd frontend && npm run dev"

echo.
echo All services started in separate windows!
echo - Frontend is running at http://localhost:3000
echo - Backend API is running at http://localhost:5000
echo.
echo Note: For a showcase, you don't need Docker, Redis, or Celery.
echo Close the popup command prompt windows to stop the servers when done.
pause
