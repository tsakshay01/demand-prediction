@echo off
TITLE DemandAI Launcher
echo ===================================================
echo   Starting DemandAI (Local Mode)
echo ===================================================

echo [1/3] Starting ML Service (Python)...
:: Starts Python Flask app in a new window on Port 5000
start "DemandAI ML Service" cmd /k "cd ml_service && c:\Users\tsaks\AppData\Local\Programs\Python\Python312\python.exe app.py"

echo [2/4] Starting Web Backend (Node.js)...
:: Starts Node.js Express app in a new window on Port 3000
start "DemandAI Backend" cmd /k "echo Installing Node dependencies... && call npm install && echo Starting Backend... && node server.js"

echo [3/4] Starting React Frontend (Modern UI)...
:: Starts Vite dev server in a new window on Port 5173
start "DemandAI React Frontend" cmd /k "cd frontend && echo Installing Frontend dependencies... && call npm install && echo Starting Frontend... && npm run dev"

echo [4/4] Waiting for services to boot (20 seconds)...
timeout /t 20 /nobreak >nul

echo Launching Modern Dashboard...
start http://localhost:5173

echo.
echo SUCCESS!
echo - AI Service: http://localhost:5000
echo - Web Backend: http://localhost:3000
echo - React Frontend (Modern): http://localhost:5173
echo.
echo (Do not close the three other command windows)
pause
