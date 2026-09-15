@echo off
chcp 65001 >nul
title Режим рынка
color 0E
cd /d "%~dp0"

rem Python из окружения проекта, если оно создано; иначе системный
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

:loop
"%PYTHON%" market_regime.py
echo [!] Перезапуск через 30 сек...
timeout /t 30 /nobreak
goto loop
