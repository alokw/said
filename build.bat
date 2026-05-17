@echo off
REM Build script for saID on Windows

echo 🎙️  Building saID...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate and install dependencies
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q --upgrade pip
pip install -q pyinstaller -r requirements.txt

REM Build with PyInstaller
echo Building application...
pyinstaller build/said.spec --clean --noconfirm

echo ✅ Build complete! Check the 'dist' folder.
echo.
echo To run: dist\saID.exe
pause
