@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo Stopping Ollama...
taskkill /F /IM "ollama.exe" 2>nul

echo Stopping Neo4j...
set "NEO4J_BAT=%ROOT%Neo4j\neo4j-community-3.5.30\bin\neo4j.bat"
if exist "%NEO4J_BAT%" (
  call "%NEO4J_BAT%" stop
) else (
  echo [WARN] Neo4j.bat not found: "%NEO4J_BAT%"
)

echo Stopping Qdrant...
taskkill /F /IM "qdrant.exe" 2>nul

echo Stopping app...
taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq *app.py*" 2>nul
taskkill /F /IM "app.exe" 2>nul

echo All services stopped.
pause
endlocal