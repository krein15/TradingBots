@echo off
title Бот #4 - Wyckoff Bitget
color 0D
cd /d C:\TradingBots\Bot4_Wyckoff
:loop
python paper_trading_wyckoff.py
echo [!] Перезапуск через 30 сек...
timeout /t 30 /nobreak
goto loop
