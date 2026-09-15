@echo off
chcp 65001 >nul
title Бот #2 - Mean Reversion
color 0B
cd /d "%~dp0"

rem Python из окружения проекта, если оно создано; иначе системный
set "PYTHON=..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

:loop
"%PYTHON%" paper_trading_meanrev.py
echo [!] Перезапуск через 30 сек...
timeout /t 30 /nobreak
goto loop
