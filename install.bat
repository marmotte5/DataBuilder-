@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: DataBuilder - true one-click installer for Windows.
::
:: Strategy:
::   1. If `uv` (Astral's Rust-based Python installer) isn't installed, we
::      bootstrap it with the official PowerShell one-liner. uv lands in
::      %USERPROFILE%\.local\bin; no admin / Visual Studio required.
::   2. uv downloads a known-good Python 3.12 build (python-build-standalone)
::      so we never touch the system Python.
::   3. uv creates .venv\ and installs DataBuilder with the right extra
::      ([cuda] when an NVIDIA GPU is present, [all] otherwise).
:: ============================================================================

title DataBuilder Installer
color 0A

:: Anchor to this script's directory (Explorer launches with cwd = system32).
cd /d "%~dp0"

echo.
echo  ===============================================================
echo     DataBuilder - One-click installer
echo  ===============================================================
echo.

:: ── Update local checkout if possible ─────────────────────────────────
git pull --ff-only >nul 2>&1
if %errorlevel% equ 0 (
    echo [ok]   Repo updated to latest revision
) else (
    echo [info] Could not git-pull -- using local source
)

:: ── Make sure %USERPROFILE%\.local\bin is on PATH for THIS session ──
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

:: ── Bootstrap uv if missing ───────────────────────────────────────────
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [1/4] Installing uv ^(Astral's Python installer^) ...
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
    where uv >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo ERROR: uv install completed but binary not on PATH.
        echo Open a new Command Prompt and re-run this script.
        pause
        exit /b 1
    )
    echo        uv installed.
) else (
    echo.
    echo [1/4] uv already installed.
)

:: ── Install Python 3.12 via uv (no admin needed) ─────────────────────
echo.
echo [2/4] Ensuring Python 3.12 is available (via uv) ...
uv python install 3.12 -q
if %errorlevel% neq 0 (
    echo ERROR: uv could not install Python 3.12.
    pause
    exit /b 1
)
echo        Python 3.12 ready.

:: ── Create or reuse the project venv ──────────────────────────────────
echo.
echo [3/4] Creating .venv\ ...
if exist ".venv" (
    echo        .venv\ already exists, reusing.
) else (
    uv venv .venv --python 3.12 -q
    if !errorlevel! neq 0 (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
    echo        .venv\ created.
)
call .venv\Scripts\activate.bat

:: ── Pick the right extra (cuda vs all) ────────────────────────────────
set "EXTRA=all"
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 set "EXTRA=cuda"

echo.
echo [4/4] Installing DataBuilder with extra: .[%EXTRA%]

:: ── For CUDA: install the matching PyTorch wheel first ───────────────
if "%EXTRA%"=="cuda" (
    echo        Installing CUDA-enabled PyTorch ^(cu128 -> cu126 fallback^)...
    uv pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    if !errorlevel! neq 0 (
        echo        cu128 wheel unavailable - trying cu126
        uv pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
        if !errorlevel! neq 0 (
            echo        CUDA wheels unavailable - falling back to CPU PyTorch
            uv pip install -q torch torchvision torchaudio
        )
    )
)

uv pip install ".[%EXTRA%]"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: pip install failed.
    echo Try a smaller install:  uv pip install ".[training,export]"
    pause
    exit /b 1
)

:: ── Final sanity check ────────────────────────────────────────────────
echo.
echo Verifying the install ...
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
echo     To launch DataBuilder, double-click the new Desktop
echo     shortcut, or  run.bat  here in the project folder.
echo  ===============================================================
echo.

:: ── Create a Desktop shortcut with the DataBuilder icon ────────────
:: Skipped if anything is missing (icon, repo, etc.) so the install
:: itself never fails because of cosmetics.
if exist "dataset_sorter\assets\databuilder.ico" (
    echo Creating Desktop shortcut...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ws = New-Object -ComObject WScript.Shell;" ^
        "$lnk = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\\DataBuilder.lnk');" ^
        "$lnk.TargetPath = (Resolve-Path '%~dp0run.bat').Path;" ^
        "$lnk.WorkingDirectory = (Resolve-Path '%~dp0').Path;" ^
        "$lnk.IconLocation = (Resolve-Path '%~dp0dataset_sorter\\assets\\databuilder.ico').Path;" ^
        "$lnk.Description = 'DataBuilder - Text-to-Image Trainer';" ^
        "$lnk.Save()" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [ok]   Desktop shortcut "DataBuilder" created.
    ) else (
        echo [info] Could not create Desktop shortcut -- run.bat still works.
    )
    echo.
)

pause
