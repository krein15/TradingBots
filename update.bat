@echo off
cd /d C:\TradingBots
git add .
git commit -m "Обновление %date% %time%"
git push origin main
echo [+] GitHub обновлён!
pause