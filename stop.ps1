# stop.ps1 — Stop Railway AI System
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "Stopping Indian Railways Maintenance AI System..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object CommandLine -like "*api.py*" | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "Server stopped cleanly." -ForegroundColor Green
