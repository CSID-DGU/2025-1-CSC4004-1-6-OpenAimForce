@echo off
setlocal

REM Check if vcpkg.exe exists
if not exist vcpkg\vcpkg.exe (
    echo vcpkg not found, setting up...
    call setup_vcpkg.bat
)

REM Install packages
cd vcpkg
.\vcpkg install ixwebsocket --triplet x86-windows --recurse
.\vcpkg install poco[netssl,crypto] --triplet x86-windows --recurse

endlocal
pause
