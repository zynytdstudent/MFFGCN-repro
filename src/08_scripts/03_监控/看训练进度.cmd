@echo off
chcp 65001 >nul
cd /d "%~dp0"
python status_all.py %1
echo.
pause
