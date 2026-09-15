@echo off
echo ============================================================
echo Starting Indian Railways Maintenance AI System...
echo ============================================================

start "" python api.py

timeout /t 3 /nobreak > NUL

echo Opening Web Operations Dashboard at http://localhost:8000/ ...
start http://localhost:8000/

echo Server is running! Run stop.bat to shut down.
