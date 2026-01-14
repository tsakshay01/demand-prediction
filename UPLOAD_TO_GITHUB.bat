@echo off
echo ========================================================
echo       DEMAND PREDICTION SYSTEM - GITHUB UPLOADER
echo ========================================================
echo.
echo Step 1: Go to https://github.com/new and create a repository.
echo         Name it 'demand-prediction'.
echo.
set REPO_URL=https://github.com/tsakshay01/demand-prediction.git

echo.
echo Connecting to GitHub...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo.
echo Uploading files...
git branch -M main
git push -u origin main

echo.
echo ========================================================
echo                 UPLOAD COMPLETE!
echo ========================================================
pause
exit

:error
echo.
echo Error: You didn't paste the URL!
pause
