@echo off
TITLE DemandAI Launcher
echo ===================================================
echo   Starting DemandAI (Local Mode)
echo ===================================================

echo [1/3] Starting ML Service (Python)...
:: Starts Python Flask app in a new window on Port 5000
start "DemandAI ML Service" cmd /k "cd ml_service && c:\Users\tsaks\AppData\Local\Programs\Python\Python312\python.exe app.py"

echo [2/3] Starting Web Backend (Node.js)...
:: Starts Node.js Express app in a new window on Port 3000
:: using 'call npm install' just in case dependencies are missing
start "DemandAI Backend" cmd /k "echo Installing dependencies... && call npm install && echo Starting Server... && node server.js"

echo [3/3] Waiting for services to boot (10 seconds)...
timeout /t 10 /nobreak >nul

echo Launching Dashboard...
start http://localhost:3000

echo.
echo SUCCESS!
echo - ML Service running on http://localhost:5000
echo - Web App running on http://localhost:3000
echo.
echo (Do not close the two other command windows)
pause
