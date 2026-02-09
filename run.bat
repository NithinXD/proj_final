@echo off
echo Starting Tamil PDF QA System...
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Check if activation was successful
if errorlevel 1 (
    echo Error: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Start Streamlit app
streamlit run app.py

pause
