@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [docker-build] Сборка образа vLLM в docker/
docker compose build

if %ERRORLEVEL% neq 0 (
  echo [docker-build] Ошибка сборки. Проверьте, что Docker запущен и Dockerfile в папке docker/
  exit /b 1
)

echo [docker-build] Готово. Запуск: docker-compose-up.bat
exit /b 0
