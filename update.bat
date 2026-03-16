@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: DataBuilder - Updater (Windows)
::
:: Reinstalls / upgrades PyTorch and all dependencies inside the existing venv.
:: Run this when you get DLL errors, want a newer PyTorch, or after a
:: driver update.
::
:: Requires: venv/ created by install.bat
:: ============================================================================

title DataBuilder Updater
color 0B

echo.
echo  =============================================================
echo     DataBuilder - Updater
echo     Reinstall / Upgrade Dependencies
echo  =============================================================
echo.

:: ── Check venv ──────────────────────────────────────────────────────

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo        Run install.bat first to create it.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

:: ── Upgrade pip ─────────────────────────────────────────────────────

echo [1/5] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo        pip upgraded.

:: ── Reinstall PyTorch ───────────────────────────────────────────────

echo.
echo [2/5] Reinstalling PyTorch with CUDA 12.8...
echo        Uninstalling old version first...
pip uninstall -y torch torchvision torchaudio >nul 2>&1

echo        Installing fresh PyTorch (this may take a few minutes)...
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

:: ── Upgrade core dependencies ───────────────────────────────────────

echo.
echo [3/5] Upgrading core dependencies...
pip install --upgrade PyQt6>=6.5 numpy>=1.24 Pillow>=10.0
if %errorlevel% neq 0 (
    echo WARNING: Some core dependencies failed to upgrade.
)

:: ── Upgrade training dependencies ───────────────────────────────────

echo.
echo [4/5] Upgrading training dependencies...
pip install --upgrade diffusers>=0.28 transformers>=4.38 accelerate>=0.27 safetensors>=0.4 peft>=0.10
if %errorlevel% neq 0 (
    echo WARNING: Some training dependencies failed to upgrade.
)

:: Optional optimizers
echo.
echo [4b/5] Upgrading optional optimizers...
pip install --upgrade bitsandbytes >nul 2>&1 && echo        bitsandbytes - OK || echo        bitsandbytes - skipped
pip install --upgrade prodigyopt >nul 2>&1 && echo        prodigyopt - OK || echo        prodigyopt - skipped
pip install --upgrade lion-pytorch >nul 2>&1 && echo        lion-pytorch - OK || echo        lion-pytorch - skipped
pip install --upgrade dadaptation >nul 2>&1 && echo        dadaptation - OK || echo        dadaptation - skipped

:: ── Verify installation ─────────────────────────────────────────────

echo.
echo [5/5] Verifying installation...
echo.

python -c "import torch; print(f'  PyTorch:     {torch.__version__}'); print(f'  CUDA:        {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}'); print(f'  GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'  VRAM:        {round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1)} GB' if torch.cuda.is_available() else ''); print(f'  bf16:        {torch.cuda.is_bf16_supported()}' if torch.cuda.is_available() else ''); print(f'  cuDNN:       {torch.backends.cudnn.version()}' if torch.backends.cudnn.is_available() else '')"

if %errorlevel% neq 0 (
    echo.
    echo WARNING: PyTorch verification failed. Try the following:
    echo   1. Update your NVIDIA drivers from https://www.nvidia.com/drivers
    echo   2. Install Visual C++ Redistributable from https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo   3. Re-run this updater
    echo.
)

python -c "import diffusers; print(f'  Diffusers:   {diffusers.__version__}')" 2>nul
python -c "import transformers; print(f'  Transformers:{transformers.__version__}')" 2>nul
python -c "import peft; print(f'  PEFT:        {peft.__version__}')" 2>nul
python -c "import PyQt6.QtCore; print(f'  PyQt6:       {PyQt6.QtCore.PYQT_VERSION_STR}')" 2>nul

echo.
echo  =============================================================
echo     Update complete!
echo.
echo     If you still get DLL errors, make sure to:
echo       1. Update NVIDIA drivers
echo       2. Install VC++ Redistributable (x64)
echo       3. Reboot your PC after driver updates
echo.
echo     To run DataBuilder:  double-click run.bat
echo  =============================================================
echo.

pause
