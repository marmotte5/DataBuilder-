@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: DataBuilder - Updater (Windows)
::
:: Diagnoses and fixes PyTorch / DLL issues, then reinstalls dependencies.
:: Run this when you get c10.dll or other DLL errors.
::
:: Requires: venv/ created by install.bat
:: ============================================================================

title DataBuilder Updater
color 0B

echo.
echo  =============================================================
echo     DataBuilder - Updater ^& Diagnostic
echo     Pull latest code, fix DLL errors, reinstall PyTorch
echo  =============================================================
echo.

:: ── Step 0: Pull latest code ──────────────────────────────────────────

echo [0/7] Pulling latest code from git...
git --version >nul 2>&1
if %errorlevel% equ 0 (
    git pull 2>nul
    if !errorlevel! equ 0 (
        echo        Code updated.
    ) else (
        echo        Git pull failed ^(offline or no remote^). Continuing with local files.
    )
) else (
    echo        Git not found. Skipping code update.
)
echo.

:: ── Step 1: Diagnostics ─────────────────────────────────────────────

echo [1/7] Running diagnostics...
echo.

:: Check Python source (Microsoft Store Python causes DLL issues)
for /f "tokens=*" %%p in ('where python 2^>nul') do (
    set "PYTHON_PATH=%%p"
    goto :found_python
)
echo ERROR: Python not found in PATH.
pause
exit /b 1

:found_python
echo   Python path:  %PYTHON_PATH%
echo %PYTHON_PATH% | findstr /i "WindowsApps" >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo   *** WARNING: You are using Microsoft Store Python! ***
    echo   Microsoft Store Python sandboxes DLL loading and causes c10.dll errors.
    echo   Please uninstall it and install Python from https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   Python ver:   %PYVER%

:: Check VC++ Redistributable
set "VCREDIST_FOUND=0"
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64" /v Version >nul 2>&1
if %errorlevel% equ 0 set "VCREDIST_FOUND=1"
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\X64" >nul 2>&1
if %errorlevel% equ 0 set "VCREDIST_FOUND=1"

if "%VCREDIST_FOUND%"=="0" (
    echo   VC++ Redist:  NOT FOUND
    echo.
    echo   *** Visual C++ Redistributable is MISSING! ***
    echo   This is the most common cause of c10.dll errors.
    echo   Downloading and installing now...
    echo.
    curl -L -o "%TEMP%\vc_redist.x64.exe" "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    if exist "%TEMP%\vc_redist.x64.exe" (
        echo   Running VC++ Redistributable installer...
        "%TEMP%\vc_redist.x64.exe" /install /passive /norestart
        echo   VC++ Redistributable installed.
        del "%TEMP%\vc_redist.x64.exe" >nul 2>&1
    ) else (
        echo   Download failed. Please install manually:
        echo   https://aka.ms/vs/17/release/vc_redist.x64.exe
    )
) else (
    echo   VC++ Redist:  Installed
)

:: Check NVIDIA driver
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%v in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do (
        echo   NVIDIA driver: %%v
    )
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
        echo   GPU:           %%g
    )
    for /f "tokens=*" %%m in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader 2^>nul') do (
        echo   VRAM:          %%m
    )
) else (
    echo   NVIDIA driver: nvidia-smi not found (driver may be too old)
)

echo.

:: ── Step 2: Activate venv ───────────────────────────────────────────

echo [2/7] Activating virtual environment...
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
echo        venv activated.

:: ── Step 3: Upgrade pip ─────────────────────────────────────────────

echo.
echo [3/7] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo        pip upgraded.

:: ── Step 4: Reinstall PyTorch ───────────────────────────────────────

echo.
echo [4/7] Reinstalling PyTorch with CUDA 12.8...
echo        Uninstalling old version first...
pip uninstall -y torch torchvision torchaudio >nul 2>&1

:: Also clean up any leftover torch files that can cause stale DLL issues
if exist "venv\Lib\site-packages\torch" (
    echo        Cleaning leftover torch files...
    rmdir /s /q "venv\Lib\site-packages\torch" >nul 2>&1
)

echo        Installing fresh PyTorch (this may take a few minutes)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if !errorlevel! neq 0 (
    echo.
    echo WARNING: CUDA 12.8 PyTorch failed. Trying CUDA 12.6...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    if !errorlevel! neq 0 (
        echo.
        echo WARNING: CUDA 12.6 failed too. Installing CPU-only PyTorch...
        pip install torch torchvision torchaudio
    )
)

:: ── Step 5: Upgrade core dependencies ───────────────────────────────

echo.
echo [5/7] Upgrading core dependencies...
pip install --upgrade "PyQt6>=6.5" "numpy>=1.24" "Pillow>=10.0"
if %errorlevel% neq 0 (
    echo WARNING: Some core dependencies failed to upgrade.
)

:: ── Step 6: Upgrade training dependencies ───────────────────────────

echo.
echo [6/7] Upgrading training dependencies...
pip install --upgrade "diffusers>=0.28" "transformers>=4.38" "accelerate>=0.27" "safetensors>=0.4" "peft>=0.10"
if %errorlevel% neq 0 (
    echo WARNING: Some training dependencies failed to upgrade.
)

:: Verify from_single_file support (needed for .safetensors model loading)
python -c "from diffusers import DiffusionPipeline; assert hasattr(DiffusionPipeline, 'from_single_file')" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo        WARNING: diffusers from_single_file not available.
    echo        Force-upgrading diffusers for .safetensors model support...
    pip install --upgrade --force-reinstall "diffusers>=0.28"
)

:: Optional optimizers
echo.
echo [6b/7] Upgrading optional optimizers...
pip install --upgrade bitsandbytes >nul 2>&1 && echo        bitsandbytes - OK || echo        bitsandbytes - skipped
pip install --upgrade prodigyopt >nul 2>&1 && echo        prodigyopt - OK || echo        prodigyopt - skipped
pip install --upgrade lion-pytorch >nul 2>&1 && echo        lion-pytorch - OK || echo        lion-pytorch - skipped
pip install --upgrade dadaptation >nul 2>&1 && echo        dadaptation - OK || echo        dadaptation - skipped
pip install --upgrade came-pytorch >nul 2>&1 && echo        came-pytorch - OK || echo        came-pytorch - skipped
pip install --upgrade schedulefree >nul 2>&1 && echo        schedulefree - OK || echo        schedulefree - skipped
pip install --upgrade triton-windows >nul 2>&1 && echo        triton-windows - OK || echo        triton-windows - skipped

:: ── Step 7: Verify PyTorch ──────────────────────────────────────────

echo.
echo [7/7] Verifying PyTorch...
echo.

python -c "import torch; print(f'  PyTorch:     {torch.__version__}'); print(f'  CUDA:        {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}'); print(f'  GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'  VRAM:        {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)} GB' if torch.cuda.is_available() else ''); print(f'  bf16:        {torch.cuda.is_bf16_supported()}' if torch.cuda.is_available() else ''); print(f'  cuDNN:       {torch.backends.cudnn.version()}' if torch.backends.cudnn.is_available() else '')"

if %errorlevel% neq 0 (
    echo.
    echo  *** PyTorch still fails to load! ***
    echo.
    echo  Troubleshooting checklist:
    echo    1. Make sure Python is from python.org (NOT Microsoft Store)
    echo    2. Reboot your PC (needed after VC++ / driver installs)
    echo    3. Temporarily disable antivirus and try again
    echo    4. If on Python 3.13+, try Python 3.11 or 3.12 instead
    echo    5. Try CPU-only:  pip install torch torchvision torchaudio
    echo.
) else (
    echo.
    echo  PyTorch loaded successfully!
)

python -c "import diffusers; print(f'  Diffusers:   {diffusers.__version__}')" 2>nul
python -c "from diffusers import DiffusionPipeline; ok='Yes' if hasattr(DiffusionPipeline,'from_single_file') else 'NO - .safetensors loading will fail!'; print(f'  SingleFile:  {ok}')" 2>nul
python -c "import transformers; print(f'  Transformers:{transformers.__version__}')" 2>nul
python -c "import peft; print(f'  PEFT:        {peft.__version__}')" 2>nul
python -c "import PyQt6.QtCore; print(f'  PyQt6:       {PyQt6.QtCore.PYQT_VERSION_STR}')" 2>nul

echo.
echo  =============================================================
echo     Update complete!
echo.
echo     To run DataBuilder:  double-click run.bat
echo  =============================================================
echo.

pause
