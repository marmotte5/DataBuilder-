@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: DataBuilder - One-Click Installer (Windows)
::
:: Installs Python environment, CUDA 12.8 PyTorch, and all dependencies
:: for the Dataset Sorter + Trainer application.
::
:: Requires: Python 3.10+ installed and in PATH
:: Recommended: NVIDIA GPU with 24 GB VRAM, CUDA 12.x drivers
:: ============================================================================

title DataBuilder Installer
color 0A

echo.
echo  =============================================================
echo     DataBuilder - Installer
echo     Dataset Sorter + SDXL / Z-Image Trainer
echo  =============================================================
echo.

:: ── Check Python ──────────────────────────────────────────────────────

echo [1/7] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10+ from python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo        Found Python %PYVER%

:: ── Create virtual environment ────────────────────────────────────────

echo.
echo [2/7] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo        Created venv/
) else (
    echo        venv/ already exists, reusing.
)

:: Activate venv
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

:: ── Upgrade pip ───────────────────────────────────────────────────────

echo.
echo [3/7] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo        pip upgraded.

:: ── Install PyTorch with CUDA 12.8 (latest) ──────────────────────────

echo.
echo [4/7] Installing PyTorch with CUDA 12.8 (latest stable)...
echo        This may take several minutes on first install...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo.
    echo WARNING: CUDA 12.8 PyTorch failed. Trying CUDA 12.6...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    if %errorlevel% neq 0 (
        echo.
        echo WARNING: CUDA 12.6 failed too. Installing CPU-only PyTorch...
        pip install torch torchvision torchaudio
    )
)

:: ── Install core dependencies ─────────────────────────────────────────

echo.
echo [5/7] Installing core dependencies...
pip install PyQt6>=6.5 numpy>=1.24 Pillow>=10.0
if %errorlevel% neq 0 (
    echo ERROR: Failed to install core dependencies.
    pause
    exit /b 1
)

:: ── Install training dependencies ─────────────────────────────────────

echo.
echo [6/7] Installing training dependencies...
pip install diffusers>=0.28 transformers>=4.38 accelerate>=0.27 safetensors>=0.4 peft>=0.10
if %errorlevel% neq 0 (
    echo WARNING: Some training dependencies failed. Training may not work.
)

:: Optional optimizers
echo.
echo [6b/7] Installing optional optimizers...
pip install bitsandbytes >nul 2>&1
if %errorlevel% equ 0 (
    echo        bitsandbytes (AdamW 8-bit) - OK
) else (
    echo        bitsandbytes - skipped (Linux-only or needs manual install)
)

pip install prodigyopt >nul 2>&1
if %errorlevel% equ 0 (
    echo        prodigyopt (Prodigy optimizer) - OK
) else (
    echo        prodigyopt - skipped
)

pip install lion-pytorch >nul 2>&1
if %errorlevel% equ 0 (
    echo        lion-pytorch (Lion optimizer) - OK
) else (
    echo        lion-pytorch - skipped
)

pip install dadaptation >nul 2>&1
if %errorlevel% equ 0 (
    echo        dadaptation (D-Adapt Adam) - OK
) else (
    echo        dadaptation - skipped
)

:: ── Verify installation ───────────────────────────────────────────────

echo.
echo [7/7] Verifying installation...
echo.

python -c "import torch; print(f'  PyTorch:     {torch.__version__}'); print(f'  CUDA:        {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}'); print(f'  GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'  VRAM:        {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)} GB' if torch.cuda.is_available() else ''); print(f'  bf16:        {torch.cuda.is_bf16_supported()}' if torch.cuda.is_available() else ''); print(f'  cuDNN:       {torch.backends.cudnn.version()}' if torch.backends.cudnn.is_available() else '')"

python -c "import diffusers; print(f'  Diffusers:   {diffusers.__version__}')" 2>nul
python -c "import transformers; print(f'  Transformers:{transformers.__version__}')" 2>nul
python -c "import peft; print(f'  PEFT:        {peft.__version__}')" 2>nul
python -c "import PyQt6.QtCore; print(f'  PyQt6:       {PyQt6.QtCore.PYQT_VERSION_STR}')" 2>nul

echo.
echo  =============================================================
echo     Installation complete!
echo.
echo     To run DataBuilder:
echo       1. Double-click  run.bat
echo       2. Or run:  venv\Scripts\python dataset_sorter.py
echo  =============================================================
echo.

pause
