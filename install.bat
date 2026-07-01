@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================================
REM  AnimaForge installer — creates one unified .venv with the whole stack.
REM  It finds a compatible Python (3.10/3.11); if none, it downloads a pinned
REM  standalone CPython, then hands off to scripts\bootstrap.py.
REM
REM  Pinned standalone Python (python-build-standalone). If the download 404s,
REM  bump PY_TAG/PY_FULL to a current release at:
REM    https://github.com/astral-sh/python-build-standalone/releases
REM ============================================================================
set "PY_TAG=20240814"
set "PY_FULL=3.10.14"
set "PBS_URL=https://github.com/astral-sh/python-build-standalone/releases/download/%PY_TAG%/cpython-%PY_FULL%+%PY_TAG%-x86_64-pc-windows-msvc-install_only.tar.gz"
set "RUNTIME_DIR=%~dp0python-runtime"
set "BASE_PY="

echo.
echo === AnimaForge installer ===
echo Looking for a compatible Python (3.10 or 3.11)...

call :try_python "py -3.10"
if defined BASE_PY goto have_python
call :try_python "py -3.11"
if defined BASE_PY goto have_python
call :try_python "python"
if defined BASE_PY goto have_python

echo No compatible Python found. Downloading standalone Python %PY_FULL%...
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"
set "ARCHIVE=%RUNTIME_DIR%\python.tar.gz"
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%PBS_URL%' -OutFile '%ARCHIVE%' -UseBasicParsing } catch { Write-Host $_.Exception.Message; exit 1 }"
if errorlevel 1 (
  echo.
  echo ERROR: could not download Python from:
  echo   %PBS_URL%
  echo Check your internet connection, or install Python 3.10/3.11 yourself and re-run.
  exit /b 1
)
echo Extracting...
tar -xf "%ARCHIVE%" -C "%RUNTIME_DIR%"
if errorlevel 1 ( echo ERROR: extraction failed. & exit /b 1 )
set "BASE_PY=%RUNTIME_DIR%\python\python.exe"

:have_python
echo Using base Python: %BASE_PY%
echo.
"%BASE_PY%" "%~dp0scripts\bootstrap.py"
if errorlevel 1 (
  echo.
  echo Install failed. See the messages above.
  exit /b 1
)
echo.
echo Install complete. Start the app with:  launch.bat
exit /b 0

REM ---------------------------------------------------------------------------
REM :try_python  <launcher>   — if <launcher> is a 3.10/3.11 interpreter, set
REM BASE_PY to its real executable path.
REM ---------------------------------------------------------------------------
:try_python
set "CAND="
for /f "delims=" %%P in ('%~1 -c "import sys;print(sys.executable)" 2^>nul') do set "CAND=%%P"
if not defined CAND goto :eof
%~1 -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,11) else 1)" 2>nul
if not errorlevel 1 set "BASE_PY=%CAND%"
goto :eof
