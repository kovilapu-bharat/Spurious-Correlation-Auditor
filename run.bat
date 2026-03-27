@echo off
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting ML Model Auditor...
streamlit run app.py

pause
