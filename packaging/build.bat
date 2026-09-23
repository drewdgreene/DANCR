@echo off
rem Build DANCR for Windows: a one-folder app, a zip, and (if NSIS is installed) a Setup wizard.
cd /d %~dp0\..
if not defined PY set PY=.venv\Scripts\python.exe
%PY% -c "import PyInstaller, PIL" 2>nul || %PY% -m pip install --quiet -e ".[build]" || exit /b 1
if not exist packaging\icon.ico %PY% packaging\make_icons.py || exit /b 1
%PY% -m PyInstaller --noconfirm --clean packaging\dancr.spec || exit /b 1
powershell -Command "Compress-Archive -Force -Path dist\DANCR -DestinationPath dist\DANCR-windows.zip"

for /f %%v in ('%PY% -c "import dancr;print(dancr.__version__)"') do set VERSION=%%v
where makensis >nul 2>nul
if errorlevel 1 (
  echo.
  echo   dist\DANCR\DANCR.exe        the window
  echo   dist\DANCR\dancr-cli.exe    command line and MCP
  echo   dist\DANCR-windows.zip      portable zip
  echo   NSIS ^(makensis^) was not found, so no Setup wizard was built.
  echo   Install NSIS ^(https://nsis.sourceforge.io^) and run this again for DANCR-Setup-%VERSION%.exe.
  exit /b 0
)
makensis -DAPPDIR=%CD%\dist\DANCR -DVERSION=%VERSION% -DOUTFILE=%CD%\dist\DANCR-Setup-%VERSION%.exe -DICON=%CD%\packaging\icon.ico packaging\windows\installer.nsi || exit /b 1
echo.
echo   dist\DANCR-Setup-%VERSION%.exe   installer wizard
echo   dist\DANCR-windows.zip           portable zip
