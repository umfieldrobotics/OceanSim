@echo off
REM Batch script to run UWCam_sdg.py on all YAML configs in UWCam_configs

set SCRIPT=extsUser\isaacsim.oceansim\standalone\UWCam_sdg.py
set CONFIG_DIR=extsUser\isaacsim.oceansim\standalone\UWCam_configs

set STARTTIME=%TIME%

for %%F in (%CONFIG_DIR%\*.yaml) do (
    echo Running %%F ...
    call python.bat %SCRIPT% --config "%%F" --close_on_completion
)

set ENDTIME=%TIME%

REM Calculate elapsed time
for /f "tokens=1-4 delims=:." %%a in ("%STARTTIME%") do set START_H=%%a& set START_M=%%b& set START_S=%%c& set START_MS=%%d
for /f "tokens=1-4 delims=:." %%a in ("%ENDTIME%") do set END_H=%%a& set END_M=%%b& set END_S=%%c& set END_MS=%%d

set /a START_TOTAL_MS=(%START_H%*3600 + %START_M%*60 + %START_S%)*100 + %START_MS%
set /a END_TOTAL_MS=(%END_H%*3600 + %END_M%*60 + %END_S%)*100 + %END_MS%
set /a ELAPSED_MS=%END_TOTAL_MS%-%START_TOTAL_MS%

set /a ELAPSED_S=%ELAPSED_MS% / 100
set /a ELAPSED_MS_ONLY=%ELAPSED_MS% %% 100

echo Total elapsed time: %ELAPSED_S%.%ELAPSED_MS_ONLY% seconds
pause
