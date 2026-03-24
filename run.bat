@echo off
:: Launch DataBuilder
title DataBuilder

echo Mise a jour...
git pull --ff-only 2>nul || echo Pas de connexion, lancement avec la version locale

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

python dataset_sorter.py
if %errorlevel% neq 0 (
    echo.
    echo Application exited with an error. Check the output above.
    pause
)
