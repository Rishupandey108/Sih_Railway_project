@echo off
echo ============================================================
echo Stopping Indian Railways Maintenance AI System...
echo ============================================================

powershell -Command "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object CommandLine -like '*api.py*' | Stop-Process -Force -ErrorAction SilentlyContinue"

echo Server stopped cleanly.
pause
