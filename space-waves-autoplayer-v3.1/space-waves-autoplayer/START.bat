@echo off
setlocal
cd /d "%~dp0"
echo Space Waves autoplayer - environment setup
set "BOT_ENV=.venv-compatible"
if not exist "%BOT_ENV%\Scripts\python.exe" goto select_python
"%BOT_ENV%\Scripts\python.exe" -c "import sys,struct; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 goto deps
echo Preserving an incompatible environment and creating a compatible one.
move "%BOT_ENV%" "%BOT_ENV%-backup-%RANDOM%" >nul
if errorlevel 1 goto failed
:select_python
py -3.12 -c "import struct,sys; sys.exit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 goto python312
py -3.11 -c "import struct,sys; sys.exit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 goto python311
python -c "import sys,struct; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if errorlevel 1 goto missing
python -m venv "%BOT_ENV%"
goto check_env
:python312
py -3.12 -m venv "%BOT_ENV%"
goto check_env
:python311
py -3.11 -m venv "%BOT_ENV%"
:check_env
if not exist "%BOT_ENV%\Scripts\python.exe" goto failed
:deps
if exist "%BOT_ENV%\installed-v2.1.ok" goto run
echo Installing dependencies for compatible Python. Internet is needed once.
"%BOT_ENV%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
"%BOT_ENV%\Scripts\python.exe" -c "import cv2,numpy,mss,pynput; from rapidocr_onnxruntime import RapidOCR"
if errorlevel 1 goto failed
echo ready>"%BOT_ENV%\installed-v2.1.ok"
:run
"%BOT_ENV%\Scripts\python.exe" autoplayer.py %*
pause
exit /b
:missing
echo.
echo This bot needs 64-bit Python 3.12 or 3.11. Python 3.13 and newer cannot use its OCR package.
echo Install the Windows installer (64-bit) from:
echo https://www.python.org/downloads/release/python-31210/
echo Keep your current Python installed. Install the Python launcher when offered.
echo Then close this window and run START.bat again.
echo Your old .venv folder is left untouched; this launcher uses .venv-compatible.
pause
exit /b 1
:failed
echo Setup did not complete. Read the error above; close this window before retrying.
pause
exit /b 1
