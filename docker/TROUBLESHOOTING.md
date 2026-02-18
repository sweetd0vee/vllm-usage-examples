# Устранение ошибок Docker (vLLM)

## Ошибка: `x509: certificate signed by unknown authority`

При сборке или `docker pull` появляется:

```
failed to fetch oauth token: Post "https://auth.docker.io/token": tls: failed to verify certificate: x509: certificate signed by unknown authority
```

**Причина:** Docker не доверяет сертификату при обращении к Docker Hub. Часто так бывает в корпоративной сети: прокси или фаервол подменяют HTTPS и используют свой сертификат, которого нет в хранилище доверенных корневых ЦС у Docker.

### Что сделать

1. **Добавить корневой сертификат вашей сети в доверенные**
   - Узнайте у ИТ/админов корневой CA (файл `.crt` или `.pem`), который используется для проверки HTTPS.
   - **Docker Desktop (Windows):**  
     Settings → Docker Engine → в JSON добавить (или обновить) опцию `"insecure-registries"` не нужно для этой ошибки; для доверия своему CA обычно нужно добавить сертификат в системное хранилище Windows (Trusted Root Certification Authorities) и перезапустить Docker Desktop, чтобы WSL2/гипервизор подхватили обновлённое хранилище.
   - **Linux:** скопировать `.crt` в `/usr/local/share/ca-certificates/`, затем `sudo update-ca-certificates` и перезапустить Docker (`sudo systemctl restart docker`).

2. **Проверить время и дату**
   - Неверные часы/дата на ПК ломают проверку сертификатов. Выставьте точное время (вручную или NTP).

3. **Отключить SSL-инспекцию для Docker Hub (если возможно)**
   - В настройках прокси/фаервола иногда можно исключить `*.docker.io` и `auth.docker.io` из SSL-инспекции (трафик идёт напрямую без подмены сертификата). Только по согласованию с ИТ.

4. **Скачать образ там, где нет блокировки, и перенести**
   - На машине с рабочим интернетом (например, дома):  
     `docker pull vllm/vllm-openai:latest`  
     затем `docker save -o vllm-openai.tar vllm/vllm-openai:latest`.
   - Перенесите файл `vllm-openai.tar` на нужный ПК и выполните:  
     `docker load -i vllm-openai.tar`.
   - В `docker-compose.yml` переключитесь на использование образа без сборки (см. ниже).

### Использовать готовый образ без сборки (без обращения к Docker Hub при build)

Если образ уже есть на машине (например, загружен через `docker load`), можно не собирать его из Dockerfile и не обращаться к Docker Hub при `docker compose up`:

1. В `docker-compose.yml` у сервиса `vllm` удалите секцию `build:` и оставьте только `image:`:

   ```yaml
   services:
     vllm:
       image: vllm/vllm-openai:latest
       # ... остальное без изменений
   ```

2. Запуск по-прежнему: `docker-compose-up.bat` (образ будет взят локально, без pull, если он уже есть).

Если вы перенесли образ под своим тегом (например, после `docker load`), укажите этот тег в `image:` в `docker-compose.yml`.

---

## Ошибка: `failed to copy: ... input/output error`

При `docker pull` или сохранении образа появляется что-то вроде:

```
failed to copy: failed to send write: write .../data: input/output error
```

**Причина:** ошибка записи на диск — Docker не может записать скачанные слои образа в своё хранилище (часто это диск WSL2 / виртуальный диск Docker Desktop).

### Что сделать

1. **Проверить свободное место на диске**
   - Образ vLLM занимает несколько гигабайт. Убедитесь, что на диске **C:** (и на том, где хранятся данные Docker) свободно **не менее 15–20 GB**.
   - Docker Desktop: **Settings → Resources → Disk image size** — при необходимости увеличьте лимит и нажмите **Apply & Restart**.

2. **Очистить неиспользуемые образы и кэш**
   ```bash
   docker system prune -a
   ```
   (удалит все неиспользуемые образы/контейнеры/сети; при необходимости сначала остановите нужные контейнеры.)

3. **Перезапустить Docker Desktop**
   - Полное закрытие (Quit) и повторный запуск. При проблемах с WSL2: в PowerShell `wsl --shutdown`, затем снова запустить Docker Desktop.

4. **Проверить диск на ошибки**
   - В Windows: правый клик по диску → Свойства → Сервис → Проверить (проверка диска).
   - Если диск старый или были сбои питания — возможны сбойные сектора; в тяжёлых случаях нужна замена диска.

5. **Перенести данные Docker на другой диск (если C: переполнен)**
   - Docker Desktop: **Settings → Resources → Advanced** — путь к хранилищу Docker можно изменить только при чистой установке или через экспорт/импорт данных.

6. **Использовать образ с другой машины**
   - На ПК с нормальным диском и интернетом: `docker pull vllm/vllm-openai:latest`, затем `docker save -o vllm-openai.tar vllm/vllm-openai:latest`.
   - Перенесите `vllm-openai.tar` на этот компьютер (флешка, сеть) и выполните `docker load -i vllm-openai.tar`. Так не нужно качать образ по сети на проблемный диск — запись будет один раз при `load`.
