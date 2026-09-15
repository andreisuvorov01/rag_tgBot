@echo off
rem One-time setup: creates a virtual environment on this drive (D:)
rem and installs everything there - because C: is out of space.
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
  echo venv already exists - skipping creation.
) else (
  echo Creating venv on this drive...
  python -m venv venv || (echo venv creation failed & pause & exit /b 1)
)

rem Redirect caches/temp to this drive: C: has no free space.
if not exist "rag-tmp" mkdir rag-tmp
if not exist "hf-cache" mkdir hf-cache
set TMP=%~dp0rag-tmp
set TEMP=%~dp0rag-tmp
set HF_HOME=%~dp0hf-cache
set PIP_NO_CACHE_DIR=1

echo Installing dependencies into venv (torch + sentence-transformers, may take 10-30 min)...
"venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements.txt sentence-transformers
if errorlevel 1 (
  echo Installation failed. Check the messages above.
  pause
  exit /b 1
)

rem NVIDIA GPU present -> CUDA build of torch: embeddings ~10x faster than CPU
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
  echo NVIDIA GPU detected - installing CUDA build of torch (~3 GB)...
  "venv\Scripts\python.exe" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 torch
  if errorlevel 1 echo CUDA torch install failed - CPU build stays in place, bot still works.
)

echo.
echo Done. Now start.bat will use venv automatically.
pause
