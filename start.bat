@echo off
chcp 65001 >nul
setlocal
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

rem Prefer project venv (created by setup_venv.bat) over system python
set PY=python
if exist "venv\Scripts\python.exe" set PY=venv\Scripts\python.exe

rem Keep model/tool caches on this drive (C: may be full)
set HF_HOME=%~dp0hf-cache
set TMP=%~dp0rag-tmp
set TEMP=%~dp0rag-tmp

echo === Check readiness ===
%PY% scripts\check_env.py
if errorlevel 1 (
  echo.
  echo Diagnostics found blocking problems. Fix them and run again.
  pause
  exit /b 1
)

echo.
echo === Starting bot (stop: Ctrl+C) ===
%PY% -m app.main
pause
