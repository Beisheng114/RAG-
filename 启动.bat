@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo Starting Ollama...
start "" /B ollama serve


echo Starting Neo4j...
set "NEO4J_BAT=%ROOT%Neo4j\neo4j-community-3.5.30\bin\neo4j.bat"
if exist "%NEO4J_BAT%" (
  call "%NEO4J_BAT%" start
) else (
  echo [WARN] Neo4j.bat not found: "%NEO4J_BAT%"
)

echo Starting Qdrant...
set "QDRANT_EXE=%ROOT%Qdrant\qdrant.exe"
if exist "%QDRANT_EXE%" (
  start "" /B "%QDRANT_EXE%"
) else (
  echo [WARN] Qdrant.exe not found: "%QDRANT_EXE%"
)

echo Starting app...
set "APP_EXE=%ROOT%app.exe"
if exist "%APP_EXE%" (
  start "" /B "%APP_EXE%"
) else (
  echo app.exe not found, running app.py with python...
  start "" /B python "%ROOT%app.py"
)

echo Waiting for web server at http://127.0.0.1:8002/ ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$port=8002; $deadline=(Get-Date).AddSeconds(60); while((Get-Date) -lt $deadline){ $r=Test-NetConnection -ComputerName '127.0.0.1' -Port $port -WarningAction SilentlyContinue; if($r.TcpTestSucceeded){ exit 0 }; Start-Sleep -Seconds 1 }; exit 1"

if errorlevel 1 (
  echo [WARN] Server not ready within timeout. Please check the console output/logs.
) else (
  start "" "http://127.0.0.1:8002/static/index.html"
)

endlocal

