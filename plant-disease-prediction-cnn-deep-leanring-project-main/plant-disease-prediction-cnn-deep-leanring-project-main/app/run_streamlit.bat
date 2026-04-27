@echo off
REM Run Plant Disease CNN Streamlit app using Python module (works when "streamlit" is not on PATH)
cd /d "%~dp0"
python -m streamlit run main.py --server.port 8501
pause
