@echo off
chcp 65001 >nul
rem Панель с видимой консолью — для диагностики, если что-то не запускается
cd /d "%~dp0\.."
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" Panel\server.py
pause
