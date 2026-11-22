@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cd /d "%~dp0"
if not exist build_abl mkdir build_abl
for %%M in (0 1 2) do (
  nvcc -O2 --optix-ir -arch=compute_75 -std=c++17 --use_fast_math -DABL_MODE=%%M -I. -I..\third_party\optix\include -o build_abl\abl_m%%M.ir ablPrograms.cu || exit /b 1
  nvcc -O2 --optix-ir -arch=compute_75 -std=c++17 --use_fast_math -DABL_MODE=%%M -DABL_INSTR -I. -I..\third_party\optix\include -o build_abl\abl_m%%M_instr.ir ablPrograms.cu || exit /b 1
  bin2c --const --name abl_m%%M build_abl\abl_m%%M.ir > build_abl\abl_m%%M.h || exit /b 1
  bin2c --const --name abl_m%%M_instr build_abl\abl_m%%M_instr.ir > build_abl\abl_m%%M_instr.h || exit /b 1
)
nvcc -O2 -arch=sm_120 -std=c++17 --use_fast_math -I. -Ibuild_abl -I..\third_party\optix\include -o build_abl\abl.exe abl.cu advapi32.lib cfgmgr32.lib || exit /b 1
nvcc -O2 -arch=sm_120 -std=c++17 --use_fast_math -DABL_INSTR -I. -Ibuild_abl -I..\third_party\optix\include -o build_abl\abl_instr.exe abl.cu advapi32.lib cfgmgr32.lib
exit /b %errorlevel%
