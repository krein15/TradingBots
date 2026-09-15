@echo off
chcp 65001 >nul
title Бот #5 - Дончиан 4ч (Bitget перпетуалы)
color 0F
cd /d "%~dp0"

rem Python из окружения проекта, если оно создано; иначе системный
set "PYTHON=..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

:loop
"%PYTHON%" paper_trading_donchian.py
echo [!] Перезапуск через 30 сек...
timeout /t 30 /nobreak
goto loop
