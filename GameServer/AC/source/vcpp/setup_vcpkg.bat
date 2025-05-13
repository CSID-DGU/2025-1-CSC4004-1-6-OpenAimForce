@echo off
setlocal

REM Check if VCPKG_ROOT is already defined
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v VCPKG_ROOT 2^>nul') do (
    echo Error: System environment variable VCPKG_ROOT is already set to %%b
    exit /b 1
)

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo Git not found. Installing Git via winget...
    winget install --id Git.Git -e --source winget
)

REM Clone and bootstrap vcpkg
if not exist vcpkg\NUL if not exist vcpkg\vcpkg.exe (
    echo Cloning vcpkg...
    git clone https://github.com/microsoft/vcpkg.git
) else (
    echo Error: vcpkg already exists or is already installed in .\vcpkg
    exit /b 1
)

cd vcpkg
if not exist vcpkg.exe (
    echo Bootstrapping vcpkg...
    cmd /c bootstrap-vcpkg.bat
)

REM Set system environment variable VCPKG_ROOT
setx VCPKG_ROOT "%cd%" /M
echo VCPKG_ROOT set to: %cd%

endlocal
