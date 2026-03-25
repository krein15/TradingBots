@echo off
title Режим рынка - ML Meta
color 0E
cd /d C:\TradingBots
:loop
python market_regime.py
timeout /t 30 /nobreak
goto loop
