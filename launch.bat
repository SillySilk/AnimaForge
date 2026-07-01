@echo off
cd /d "%~dp0"
REM Launch the AnimaForge GUI from the unified .venv with pythonw.exe (no console
REM window) and detach via START so this batch window closes immediately.
REM First time? Run install.bat to create the .venv.
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
  echo .venv not found. Run install.bat first.
  pause
  exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0main.py"

REM --- Debugging startup errors? Comment the line above and uncomment below to
REM     run with a visible console that stays open on crash:
REM "%~dp0.venv\Scripts\python.exe" "%~dp0main.py"
REM pause
