@echo off
title Бот #3 - SMC v2 Bitget
color 0E
cd /d C:\TradingBots\Bot3_SMC
:loop
python paper_trading_smc_v2.py
echo [!] Перезапуск через 30 сек...
timeout /t 30 /nobreak
goto loop
