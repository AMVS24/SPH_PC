@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cd /d "%~dp0\.."
if not exist bench\fp64\build mkdir bench\fp64\build
nvcc -O2 -arch=sm_120 -std=c++17 -DGPU_DOUBLE -I. -Ibackend -shared -o bench\fp64\build\cudaPhysics.dll cudaPhysics.cu
if errorlevel 1 exit /b 1
copy /y bench\build\bench.exe bench\fp64\build\bench.exe >nul
exit /b %errorlevel%
