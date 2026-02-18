# vLLM Usage Examples

Учебный проект с примерами использования **[vLLM](https://vllm.ai)** — высокопроизводительной системы инференса и сервинга больших языковых моделей (LLM). Подходит для знакомства с офлайн-инференсом, OpenAI-совместимым API и запуском в Docker.

---

## О проекте

vLLM даёт два основных способа работы с LLM:

| Режим | Описание |
|-------|----------|
| **Офлайн-инференс** | Класс `LLM` в Python: загрузка модели и пакетная генерация через `generate()` / `chat()`. |
| **Онлайн-сервинг** | Сервер с OpenAI-совместимым API для чат-ботов, RAG и клиентов под OpenAI. |

В этом репозитории — готовые примеры под оба режима, конфигурация для Docker и краткая документация на русском.

---

## Требования

- **Python 3.10–3.13**
- **ОС:** Linux (полная поддержка GPU); на Windows — WSL2 или Docker (нативный vLLM под Windows не поддерживается)
- **GPU** (рекомендуется): NVIDIA CUDA, AMD ROCm или Intel XPU; для маленьких моделей возможен CPU
- Для Docker: [Docker](https://docs.docker.com/get-docker/) и при использовании GPU — [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

---

## Структура проекта

```
vllm-usage-examples/
├── README.md                 # этот файл
├── VLLM_DOCUMENTATION_RU.md   # документация по vLLM (установка, режимы, Docker)
├── examples/                  # Python-примеры
│   ├── README.md              # описание примеров
│   ├── 01_offline_basic.py    # базовый пакетный инференс
│   ├── 02_offline_chat.py     # чат через llm.chat()
│   ├── 03_sampling_params.py  # параметры сэмплирования
│   ├── 04_client_openai.py   # клиент к API (completions, chat, streaming)
│   ├── 05_batch_evaluation.py # пакетная обработка датасета
│   ├── 06_rag_simple.py      # простой RAG (контекст + вопрос)
│   └── 07_offline_with_chat_template.py  # chat template через transformers
└── docker/                    # запуск vLLM в контейнере
    ├── docker-compose.yml     # сервис vLLM (GPU, порт 8000)
    ├── docker-compose-up.bat  # запуск
    ├── docker-compose-down.bat # остановка
    ├── .env.example           # VLLM_MODEL, HF_TOKEN
    └── TROUBLESHOOTING.md     # частые ошибки (сертификаты, диск)
```

---

## Быстрый старт

### Вариант 1: Локальная установка (Linux / WSL2)

```bash
# Виртуальное окружение
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Установка vLLM (NVIDIA)
pip install vllm

# Дополнительно для примеров с API и chat template
pip install openai transformers
```

Запуск офлайн-примера:

```bash
python examples/01_offline_basic.py
```

Запуск сервера (в отдельном терминале):

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

Затем клиент к API:

```bash
python examples/04_client_openai.py
```

### Вариант 2: Docker (в т.ч. Windows)

1. Убедитесь, что образ есть локально (или выполните на машине с интернетом `docker pull vllm/vllm-openai:latest`, затем `docker save` / `docker load` на целевую — см. [docker/TROUBLESHOOTING.md](docker/TROUBLESHOOTING.md)).
2. В папке `docker/` при необходимости скопируйте `.env.example` в `.env` и задайте `VLLM_MODEL` и `HF_TOKEN`.
3. Запуск и остановка:

   ```bat
   docker\docker-compose-up.bat
   docker\docker-compose-down.bat
   ```

API будет доступен по адресу **http://localhost:8000** (проверка: `curl http://localhost:8000/v1/models`).

---

## Примеры (`examples/`)

| Файл | Описание |
|------|----------|
| **01_offline_basic.py** | Базовый пакетный инференс: список промптов → `llm.generate()`. |
| **02_offline_chat.py** | Чат-модели через `llm.chat()` (system/user/assistant). |
| **03_sampling_params.py** | Параметры: temperature, top_p, top_k, repetition_penalty и др. |
| **04_client_openai.py** | Клиент к vLLM как к OpenAI API: models, completions, chat, streaming. |
| **05_batch_evaluation.py** | Пакетная обработка датасета (оценка, массовая генерация). |
| **06_rag_simple.py** | Простой RAG: контекст и вопрос в одном промпте. |
| **07_offline_with_chat_template.py** | Ручное применение chat template через `transformers`. |

Подробнее — в [examples/README.md](examples/README.md).

---

## Модели

В примерах используются:

- **facebook/opt-125m** — маленькая модель для быстрой проверки (слабый GPU / CPU).
- **Qwen/Qwen2.5-1.5B-Instruct** — лёгкая chat-модель; при достаточной памяти можно заменить на 7B/72B.

Идентификатор модели можно менять в коде или через переменные окружения / `.env` (например, `VLLM_MODEL` для Docker).

---

## Документация и ссылки

- **[VLLM_DOCUMENTATION_RU.md](VLLM_DOCUMENTATION_RU.md)** — установка, режимы работы, запуск в Docker, шпаргалка команд.
- **[docker/TROUBLESHOOTING.md](docker/TROUBLESHOOTING.md)** — ошибки при работе с Docker (сертификаты, нехватка места на диске).
- [Официальная документация vLLM](https://docs.vllm.ai)
- [Поддерживаемые модели](https://docs.vllm.ai/en/stable/models/supported_models/)
- [Репозиторий vLLM](https://github.com/vllm-project/vllm)

---

## Лицензия

Примеры и текст документации в репозитории — для обучения и свободного использования. Сам vLLM и используемые модели подчиняются своим лицензиям (Apache 2.0 для vLLM; у каждой модели на Hugging Face — своя лицензия).
