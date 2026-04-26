@echo off
:: ============================================================================
:: DataBuilder - Windows double-click launcher.
::
:: Activates .venv\ and runs `python -m dataset_sorter`. Designed to be
:: double-clicked from Explorer.
:: ============================================================================

title DataBuilder

:: Cd into the directory this .bat file lives in (Explorer launches with
:: cwd = something else, depending on the user's shortcut).
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo  No .venv\ here -- double-click install.bat first.
    echo.
    pause
    exit /b 1
)

:: Silent auto-update; ignore failures (offline).
git pull --ff-only >nul 2>&1

call .venv\Scripts\activate.bat

python -m dataset_sorter
if %errorlevel% neq 0 (
    echo.
    echo DataBuilder exited with error code %errorlevel%.
    pause
)
