@echo off
rem In a venv Python is beside this launcher; in a standard Windows install
rem it is in the parent of Scripts. Use that interpreter, not one on PATH.
if exist "%~dp0python.exe" (
    "%~dp0python.exe" -m lplearn %*
) else if exist "%~dp0..\python.exe" (
    "%~dp0..\python.exe" -m lplearn %*
) else (
    echo lplearn: Python interpreter was not found beside the Scripts directory 1>&2
    exit /b 1
)
