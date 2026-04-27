@echo off
echo ========================================
echo    Vision Transformer Plant Disease
echo    Detection System
echo ========================================
echo.
echo 🤖 Starting Vision Transformer (ViT)...
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    echo 📥 Downloading Python installer...
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found

REM Check if required packages are installed
echo 📦 Checking Vision Transformer dependencies...
python -c "import torch, torchvision, streamlit, PIL, transformers" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📥 Installing Vision Transformer requirements...
    pip install -r vit_requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Vision Transformer dependencies installed
)

echo.
echo 🚀 Launching Vision Transformer Plant Disease Detection...
echo 📱 Opening web browser...
echo.

REM Start Streamlit app
streamlit run vit_plant_disease_app.py --server.port 8502

if %errorlevel% neq 0 (
    echo ❌ Failed to start Vision Transformer app
    pause
    exit /b 1
)

echo.
echo ✅ Vision Transformer app stopped
pause
