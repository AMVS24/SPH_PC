@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cd /d "%~dp0"
if not exist build_cost mkdir build_cost
nvcc -O2 --optix-ir -arch=compute_75 -std=c++17 --use_fast_math -I. -I..\third_party\optix\include -o build_cost\cost_programs.ir costPrograms.cu
if errorlevel 1 exit /b 1
nvcc -O2 --optix-ir -arch=compute_75 -std=c++17 --use_fast_math -DCOST_INSTR -I. -I..\third_party\optix\include -o build_cost\cost_programs_instr.ir costPrograms.cu
if errorlevel 1 exit /b 1
bin2c --const --name cost_programs build_cost\cost_programs.ir > build_cost\cost_programs.h
if errorlevel 1 exit /b 1
bin2c --const --name cost_programs_instr build_cost\cost_programs_instr.ir > build_cost\cost_programs_instr.h
if errorlevel 1 exit /b 1
nvcc -O2 -arch=sm_120 -std=c++17 -I. -Ibuild_cost -I..\third_party\optix\include -o build_cost\cost.exe cost.cu advapi32.lib cfgmgr32.lib
if errorlevel 1 exit /b 1
nvcc -O2 -arch=sm_120 -std=c++17 -DCOST_INSTR -I. -Ibuild_cost -I..\third_party\optix\include -o build_cost\cost_instr.exe cost.cu advapi32.lib cfgmgr32.lib
exit /b %errorlevel%
