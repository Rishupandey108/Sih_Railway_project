# start.ps1 — Start Railway AI System
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting Indian Railways Maintenance AI System..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Start-Process -FilePath "python" -ArgumentList "api.py" -WindowStyle Normal

Start-Sleep -Seconds 3

Write-Host "Opening Web Dashboard: http://localhost:8000/" -ForegroundColor Green
Start-Process "http://localhost:8000/"
