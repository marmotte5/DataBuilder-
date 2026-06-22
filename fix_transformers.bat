@echo off
:: ============================================================================
:: DataBuilder - one-click fix for the "transformers too new" error.
::
:: transformers 5.x removed an internal CLIP attribute that diffusers'
:: single-file checkpoint loader still needs, so loading a .safetensors
:: SDXL/SD1.5 model fails. This pins transformers back to the 4.x line
:: inside the project's .venv. Double-click it once, then relaunch run.bat.
:: ============================================================================

title DataBuilder - Fix transformers
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo  No .venv\ here -- double-click install.bat first.
    echo.
    pause
    exit /b 1
)

:: uv lands in the user's .local\bin after install.bat; make sure it's findable.
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

call .venv\Scripts\activate.bat

echo.
echo Pinning transformers to the 4.x line (needed for single-file checkpoints)...
echo.
uv pip install "transformers<5"
if %errorlevel% neq 0 (
    echo uv not found -- trying pip instead...
    python -m pip install "transformers<5"
)

echo.
echo Verifying:
python -c "import transformers; print('  transformers: ' + transformers.__version__)"
echo.
echo Done. You can now relaunch DataBuilder with run.bat.
echo.
pause
