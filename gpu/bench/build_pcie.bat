@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cd /d "%~dp0"
if not exist build mkdir build
nvcc -O2 -arch=sm_120 -o build\pcie_bench.exe pcie_bench.cu
exit /b %errorlevel%
