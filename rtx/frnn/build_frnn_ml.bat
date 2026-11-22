@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cd /d "%~dp0"
if not exist build mkdir build
nvcc -O2 --optix-ir -arch=compute_75 -std=c++17 --use_fast_math -I. -I..\third_party\optix\include -o build\frnn_programs.ir frnnPrograms.cu
if errorlevel 1 exit /b 1
bin2c --const --name frnn_programs build\frnn_programs.ir > build\frnn_programs.h
if errorlevel 1 exit /b 1
nvcc -O2 -arch=sm_120 -std=c++17 -I. -Ibuild -I..\third_party\optix\include -o build\frnn_ml.exe frnn_ml.cu advapi32.lib cfgmgr32.lib
exit /b %errorlevel%
