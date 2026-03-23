@echo off
title Статистика всех ботов
color 0F
cls

echo ================================================
echo   СТАТИСТИКА ВСЕХ БОТОВ
echo   %date% %time%
echo ================================================
echo.

echo === БОТ #1 - EMA ===
cd /d C:\TradingBots\Bot1_EMA
python paper_trading_v2_clean.py status
echo.

echo === БОТ #2 - СТРУКТУРА ===
cd /d C:\TradingBots\Bot2_Structure
python paper_trading_structure.py status
echo.

echo === БОТ #3 - SMC ===
cd /d C:\TradingBots\Bot3_SMC
python paper_trading_smc.py status
echo.

echo === БОТ #4 - WYCKOFF ===
cd /d C:\TradingBots\Bot4_Wyckoff
python paper_trading_wyckoff.py status
echo.

echo ================================================
echo   Нажми любую клавишу чтобы закрыть
echo ================================================
pause > nul
