@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [docker-compose-down] Остановка vLLM...
docker compose down

if %ERRORLEVEL% neq 0 (
  echo [docker-compose-down] Ошибка остановки контейнеров.
  exit /b 1
)

echo [docker-compose-down] Контейнер остановлен.
exit /b 0
