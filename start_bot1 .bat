@echo off
title Бот #1 - EMA
color 0A
cd /d C:\TradingBots\Bot1_EMA
set PYTHON=C:\Users\reink\AppData\Local\Programs\Python\Python314\python.exe
:loop
%PYTHON% paper_trading_v2_clean.py
echo [!] Перезапуск через 30 сек...
timeout /t 30 /nobreak
goto loop
