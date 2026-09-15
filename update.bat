@echo off
chcp 65001 >nul
title Отправка на GitHub
cd /d "%~dp0"

rem Сообщение коммита — аргументом:  update.bat "что изменилось"
rem Без аргумента спросим, чтобы в истории не копились
rem безымянные "Обновление <дата>".
set "MSG=%~1"
if "%MSG%"=="" set /p "MSG=Что изменилось: "
if "%MSG%"=="" (
    echo [!] Пустое сообщение — отменено.
    pause
    exit /b 1
)

git add .
git commit -m "%MSG%"
git push origin HEAD
echo [+] Отправлено на GitHub
pause
