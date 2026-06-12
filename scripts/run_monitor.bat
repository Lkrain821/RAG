@echo off
cd /d D:\Pythoncode\RAG
call "D:\Pythoncode\RAG\venv\Scripts\python.exe" "D:\Pythoncode\RAG\scripts\monitor.py" > "D:\Pythoncode\RAG\.monitor_last_run.log" 2>&1
