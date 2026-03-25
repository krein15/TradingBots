@echo off
title Бот #2 - Mean Reversion
color 0B
cd /d C:\TradingBots\Bot2_MeanRev
:loop
python paper_trading_meanrev.py
echo [!] Перезапуск через 30 сек...
timeout /t 30 /nobreak
goto loop
