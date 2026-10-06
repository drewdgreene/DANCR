@echo off
rem Bundle MinerU beside a Windows DANCR build, so documents work out of the box.
rem   packaging\build-mineru.bat [DIST_DIR]      (default: dist\DANCR)
rem   set MINERU_TIER=standard                   (default: basic)
rem Creates a RELOCATABLE Python 3.12 venv with MinerU and the requested models at <DIST_DIR>\.dancr-mineru.
setlocal
cd /d %~dp0\..
if "%~1"=="" (set DEST=dist\DANCR) else (set DEST=%~1)
if not defined MINERU_TIER set MINERU_TIER=basic
set ENV=%DEST%\.dancr-mineru

where uv >nul 2>nul || (echo build-mineru needs uv ^(https://docs.astral.sh/uv^) & exit /b 1)
if not exist "%DEST%" (echo Build DANCR first: %DEST% not found ^(run packaging\build.bat^) & exit /b 1)

echo Creating a relocatable MinerU environment at %ENV% (Python 3.12)...
if exist "%ENV%" rmdir /s /q "%ENV%"
uv venv --relocatable --python 3.12 "%ENV%" || exit /b 1
uv pip install --python "%ENV%\Scripts\python.exe" "mineru>=4.0,<5" || exit /b 1

echo Downloading MinerU '%MINERU_TIER%' models into the bundle...
set MINERU_HOME=%CD%\%ENV%
"%ENV%\Scripts\mineru-models-download.exe" --tier %MINERU_TIER% || echo Model download skipped or failed; MinerU will fetch models on first use.

"%ENV%\Scripts\mineru-kit.exe" --version
echo Bundled MinerU in %ENV% - the app now reads documents with no separate install.
