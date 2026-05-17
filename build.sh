#!/bin/bash
# Build script for saID
# Run this on the target platform (Mac or Windows)

set -e

echo "🎙️  Building saID..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    # Use Python 3.10+ for latest Kokoro features
    python3.11 -m venv venv || python3.10 -m venv venv || python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate || source venv/Scripts/activate

echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q pyinstaller -r requirements.txt

# Build with PyInstaller
echo "Building application..."
pyinstaller build/said.spec --clean --noconfirm

echo "✅ Build complete! Check the 'dist' folder."
echo ""
echo "To run:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  open dist/saID.app"
else
    echo "  dist\\saID.exe"
fi
