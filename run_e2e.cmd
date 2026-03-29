@echo off
cd /d "D:\学习记录\导师项目\研究\caiqiyue_file"
D:\anconda\envs\experiment\python.exe -u run_e2e.py >> e2e_output.log 2>&1
echo Exit: %ERRORLEVEL% >> e2e_output.log 2>&1
