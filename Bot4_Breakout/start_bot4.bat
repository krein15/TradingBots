@echo off
title Бот #4 - Breakout
color 0D
cd /d C:\TradingBots\Bot4_Breakout
:loop
python paper_trading_breakout.py
echo [!] Перезапуск через 30 сек...
timeout /t 30 /nobreak
goto loop
