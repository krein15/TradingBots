@echo off
title Бот #2 - Структура
color 0B
cd /d C:\TradingBots\Bot2_Structure
:loop
python paper_trading_structure_v2.py
echo [!] Перезапуск через 30 сек...
timeout /t 30 /nobreak
goto loop
