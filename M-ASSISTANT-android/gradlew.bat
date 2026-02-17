@echo off
setlocal
set SCRIPT_DIR=%~dp0
call "%SCRIPT_DIR%\gradle\wrapper\gradle-wrapper.jar" %* 2>nul || gradle %*
