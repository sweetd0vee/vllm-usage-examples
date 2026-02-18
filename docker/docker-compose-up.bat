@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [docker-compose-up] Запуск vLLM...
docker compose up -d

if %ERRORLEVEL% neq 0 (
  echo [docker-compose-up] Ошибка. Убедитесь, что образ собран: docker-build.bat
  exit /b 1
)

echo.
echo vLLM запущен. API: http://localhost:8000
echo Модели: http://localhost:8000/v1/models
echo Остановка: docker-compose-down.bat
exit /b 0
