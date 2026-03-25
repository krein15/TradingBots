@echo off
title Бот #3 - Funding Rate Bitget
color 0C
cd /d C:\TradingBots\Bot3_Funding
:loop
python paper_trading_funding.py
echo [!] Перезапуск через 30 сек...
timeout /t 30 /nobreak
goto loop
