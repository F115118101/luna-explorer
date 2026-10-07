@echo off
rem Double-click: starts the LUNA live server and opens the page in a private window.
rem Close the "LUNA server" console window to stop the server.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found in PATH.
  pause
  exit /b 1
)

set "URL=http://127.0.0.1:8000/"

rem Already running?
curl -s -o nul "%URL%health" && goto open

start "LUNA server" cmd /k python live\server.py

echo Waiting for the server (the first start loads GPT-2, ~10-30 s) ...
for /l %%i in (1,1,90) do (
  curl -s -o nul "%URL%health" && goto open
  timeout /t 2 /nobreak >nul
)
echo Server did not respond. Check the "LUNA server" window for errors.
pause
exit /b 1

:open
set "BROWSER="
set "FLAG="
for %%p in (
  "%ProgramFiles%\Google\Chrome\Application\chrome.exe"
  "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
  "%LocalAppData%\Google\Chrome\Application\chrome.exe"
) do if exist %%p if not defined BROWSER (set "BROWSER=%%~p" & set "FLAG=--incognito")

if not defined BROWSER for %%p in (
  "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
  "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
) do if exist %%p if not defined BROWSER (set "BROWSER=%%~p" & set "FLAG=-inprivate")

if defined BROWSER (
  start "" "%BROWSER%" %FLAG% "%URL%"
) else (
  start "" "%URL%"
)
exit /b 0
