@echo off
chcp 65001 >nul
REM This file lives in src\08_scripts\05_sync\ ; go up 3 levels to the project root.
cd /d "%~dp0..\..\.."

echo ========================================
echo   Pull code snapshot from zjut148
echo ========================================
echo.
echo Server src\ = flat layout (the version that actually runs)
echo Local  src\ = layered layout (reading copy, Chinese comments)
echo This script only copies server code into src\_from_server\
echo so you can diff it. Local annotated files are NOT overwritten.
echo.

if not exist "src\_from_server" mkdir "src\_from_server"

echo [1/2] Pulling Python sources...
scp -q zjut148:/public2/czz/ZhangYu/project/mffgcn/src/*.py src/_from_server/
if errorlevel 1 goto fail

echo [2/2] Pulling server-side shell scripts...
scp -q "zjut148:/public2/czz/ZhangYu/project/mffgcn/*.sh" src/_from_server/
if errorlevel 1 echo       (skipped: shell scripts not pulled, python diff still fine)

echo.
echo Done. Snapshot files:
echo ----------------------------------------
dir /b src\_from_server\*.py
echo ----------------------------------------
echo To compare: run VSCode task "1 Compare local annotated vs server"
echo.
pause
exit /b 0

:fail
echo.
echo [FAILED] Aborted. Check:
echo   1. ssh zjut148 works
echo   2. network / VPN is connected
echo.
pause
exit /b 1
