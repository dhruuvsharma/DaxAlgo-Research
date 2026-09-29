@echo off
setlocal EnableExtensions

cd /d "%~dp0"
if errorlevel 1 (
    echo Failed to open the FDT-N workspace:
    echo %~dp0
    pause
    exit /b 1
)
set "FDTN_ROOT=%CD%"

title FDT-N - DaxAlgo Docs

where codex.exe >nul 2>&1
if errorlevel 1 (
    echo Codex CLI was not found on PATH.
    echo Install or repair Codex, then run this launcher again.
    pause
    exit /b 1
)

set "FDTN_START_PROMPT=Read README.md completely. Confirm that the DaxAlgo Docs main branch and FDT-N workspace are loaded. Preserve unrelated dirty files, work only in this repository, and wait for my next task."

if /i "%~1"=="--check" (
    echo Launcher check passed.
    echo Working directory: %FDTN_ROOT%
    echo Codex profile: funkeh
    where codex.exe
    where python.exe >nul 2>&1
    if not errorlevel 1 python.exe -c "import sys; assert len(sys.argv) == 7, sys.argv; assert sys.argv[1:3] == ['--profile','funkeh'], sys.argv; assert sys.argv[3] == '-C' and sys.argv[5] == '--', sys.argv; assert 'README.md' in sys.argv[6], sys.argv; print('Profile and argument quoting: pass')" --profile funkeh -C "%FDTN_ROOT%" -- "%FDTN_START_PROMPT%"
    codex.exe --profile funkeh -C "%FDTN_ROOT%" --help >nul
    if errorlevel 1 (
        echo The funkeh profile could not be loaded.
        exit /b 1
    )
    exit /b 0
)

codex.exe --profile funkeh --dangerously-bypass-approvals-and-sandbox -C "%FDTN_ROOT%" -- "%FDTN_START_PROMPT%"
set "FDTN_EXIT=%ERRORLEVEL%"

if not "%FDTN_EXIT%"=="0" (
    echo.
    echo Codex exited with code %FDTN_EXIT%.
    pause
)

endlocal & exit /b %FDTN_EXIT%
