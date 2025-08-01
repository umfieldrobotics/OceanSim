@echo off
REM Batch script to run UWCam_sdg.py on all YAML configs in a specified config directory

REM Get the directory of this batch file
set "BASEDIR=C:\Users\mahaoyu\Code\isaacsim-4.5\"

set "SCRIPT=%BASEDIR%extsUser\isaacsim.oceansim\standalone\UWCam_sdg.py"

REM Use first argument as config directory, or default if not provided
if "%~1"=="" (
    set "CONFIG_DIR=%BASEDIR%extsUser\isaacsim.oceansim\standalone\UWCam_configs\empty"
) else (
    REM If argument is an absolute path (starts with drive letter or \\), use as is; otherwise, make it relative to BASEDIR
    echo %~1 | findstr /B /I "[A-Z]:\\ \\" >nul
    if %errorlevel%==0 (
        set "CONFIG_DIR=%~1"
    ) else (
        set "CONFIG_DIR=%BASEDIR%%~1"
    )
)

echo Using config directory: "%CONFIG_DIR%"

REM Get start time
for /f "tokens=1-4 delims=:." %%a in ("%time%") do set START=%%a%%b%%c%%d

for %%F in ("%CONFIG_DIR%\*.yaml") do (
    echo Running %%F ...
    call "%BASEDIR%python.bat" "%SCRIPT%" --config "%%F" --close_on_completion
)

REM Get end time
for /f "tokens=1-4 delims=:." %%a in ("%time%") do set END=%%a%%b%%c%%d

REM Calculate elapsed time in seconds
set /a STARTSEC=%START:~0,2%*3600 + %START:~2,2%*60 + %START:~4,2% + %START:~6,2%/100
set /a ENDSEC=%END:~0,2%*3600 + %END:~2,2%*60 + %END:~4,2% + %END:~6,2%/100
set /a ELAPSED=%ENDSEC%-%STARTSEC%

echo Total elapsed time: %ELAPSED% seconds
pause
