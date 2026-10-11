@echo off
cd /d "%~dp0"
REM Start the remote dashboard (run the batch queue from another device) hidden.
REM It listens on http://127.0.0.1:7802; reach it from elsewhere via tailscale serve.
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
  echo .venv not found. Run install.bat first.
  pause
  exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" -m remote.server
