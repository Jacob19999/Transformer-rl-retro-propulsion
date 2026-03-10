@echo off
:: run_isaac.bat — Run any Python command with Isaac Sim's environment
:: Usage: run_isaac.bat -m simulation.isaac.usd.drone_builder --config ...
::        run_isaac.bat myscript.py

setlocal

set ISAAC_DIR=C:\Users\tngzj\OneDrive\Desktop\Issac Sim
set REPO_DIR=%~dp0

set PYTHONPATH=%REPO_DIR%;%ISAAC_DIR%\extscache\omni.usd.libs-1.0.1+8131b85d.wx64.r.cp311;%PYTHONPATH%
set PATH=%ISAAC_DIR%\extscache\omni.usd.libs-1.0.1+8131b85d.wx64.r.cp311\bin;%ISAAC_DIR%\kit;%PATH%

call "%ISAAC_DIR%\python.bat" %*

endlocal
