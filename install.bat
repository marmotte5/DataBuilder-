@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: DataBuilder - One-shot installer for Windows.
::
:: What this script does:
::   1. Locates a supported Python (3.10 - 3.13). 3.14+ is rejected because
::      most ML wheels don't ship for it yet.
::   2. Creates .venv\, upgrades pip, installs DataBuilder with the right
::      extra ([cuda] when an NVIDIA GPU is present, [all] otherwise — the
::      pyproject markers make this safe across hardware now).
::   3. Prints how to launch the app.
:: ============================================================================

title DataBuilder Installer
color 0A

:: When double-clicked from Explorer, the working directory may be
:: somewhere else (system32 etc.). Anchor to this script's location.
cd /d "%~dp0"

echo.
echo  ===============================================================
echo     DataBuilder - Installer
echo  ===============================================================
echo.

:: ── Update local checkout if possible ─────────────────────────────────
git pull --ff-only >nul 2>&1
if %errorlevel% equ 0 (
    echo [ok] Repo updated to latest revision
) else (
    echo [info] Could not git-pull ^(offline or detached HEAD^) - using local source
)

:: ── Find a supported Python (3.10 - 3.13) ─────────────────────────────
echo.
echo [1/5] Looking for Python 3.10 - 3.13 ...

set "PYTHON_CMD="
set "PYTHON_VER="

for %%P in (py python python3 python3.13 python3.12 python3.11 python3.10) do (
    if not defined PYTHON_CMD (
        where %%P >nul 2>&1
        if !errorlevel! equ 0 (
            for /f "tokens=*" %%V in ('%%P -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do (
                if "%%V"=="3.10" set "PYTHON_CMD=%%P" & set "PYTHON_VER=%%V"
                if "%%V"=="3.11" set "PYTHON_CMD=%%P" & set "PYTHON_VER=%%V"
                if "%%V"=="3.12" set "PYTHON_CMD=%%P" & set "PYTHON_VER=%%V"
                if "%%V"=="3.13" set "PYTHON_CMD=%%P" & set "PYTHON_VER=%%V"
            )
        )
    )
)

if not defined PYTHON_CMD (
    echo.
    echo ERROR: No supported Python ^(3.10 - 3.13^) found in PATH.
    echo.
    echo Download from: https://www.python.org/downloads/release/python-31210/
    echo IMPORTANT: tick "Add Python to PATH" during installation.
    echo Then re-run this script.
    pause
    exit /b 1
)
echo        Found %PYTHON_CMD% --^> Python %PYTHON_VER%

:: ── Create or reuse the project venv ──────────────────────────────────
echo.
echo [2/5] Creating .venv\ ...
if exist ".venv" (
    echo        .venv\ already exists, reusing.
) else (
    %PYTHON_CMD% -m venv .venv
    if !errorlevel! neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Try installing python3-venv or download Python from python.org.
        pause
        exit /b 1
    )
    echo        .venv\ created with Python %PYTHON_VER%.
)
call .venv\Scripts\activate.bat

:: ── Upgrade pip / wheel / setuptools ──────────────────────────────────
echo.
echo [3/5] Upgrading pip, wheel, setuptools ...
python -m pip install --upgrade pip wheel setuptools -q
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)
echo        Build tooling up to date.

:: ── Pick the right extra ──────────────────────────────────────────────
set "EXTRA=all"
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    set "EXTRA=cuda"
)
echo.
echo [4/5] Installing DataBuilder with extra: .[%EXTRA%]

:: ── Install CUDA-enabled PyTorch first if NVIDIA GPU detected ─────────
if "%EXTRA%"=="cuda" (
    echo        Installing CUDA-enabled PyTorch ^(cu128, then cu126 fallback^)...
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    if !errorlevel! neq 0 (
        echo        CUDA 12.8 wheel unavailable - falling back to cu126
        pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
        if !errorlevel! neq 0 (
            echo        CUDA wheels unavailable - falling back to CPU PyTorch
            pip install -q torch torchvision torchaudio
        )
    )
)

pip install ".[%EXTRA%]"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: pip install failed.
    echo Try a smaller install:  pip install ".[training,export]"
    pause
    exit /b 1
)

:: ── Final sanity check ────────────────────────────────────────────────
echo.
echo [5/5] Verifying the install ...
python -c "import sys; print('  Python:       ' + sys.version.split()[0])"
python -c "import PyQt6.QtCore as q; print('  PyQt6:        ' + q.PYQT_VERSION_STR)" 2>nul
python -c "import torch; print('  PyTorch:      ' + torch.__version__); print('  CUDA:         ' + (torch.version.cuda if torch.cuda.is_available() else 'CPU only')); print('  GPU:          ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'))" 2>nul
python -c "import diffusers; print('  diffusers:    ' + diffusers.__version__)" 2>nul
python -c "import transformers; print('  transformers: ' + transformers.__version__)" 2>nul
python -c "import peft; print('  peft:         ' + peft.__version__)" 2>nul

echo.
echo  ===============================================================
echo     Install complete.
echo.
echo     To launch DataBuilder:
echo       .venv\Scripts\activate.bat
echo       python -m dataset_sorter
echo  ===============================================================
echo.

pause
