@echo off
chcp 65001 >nul
title Бот #4 - Breakout
color 0D
cd /d "%~dp0"

rem Python из окружения проекта, если оно создано; иначе системный
set "PYTHON=..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

:loop
"%PYTHON%" paper_trading_breakout.py
echo [!] Перезапуск через 30 сек...
timeout /t 30 /nobreak
goto loop
