# Примеры использования vLLM

Набор Python-скриптов с типовыми сценариями работы с vLLM.

## Требования

- Установленный vLLM: `pip install vllm`
- Для примеров с OpenAI-клиентом: `pip install openai`
- Для `07_offline_with_chat_template.py`: `pip install transformers`
- GPU рекомендуется (для больших моделей обязателен)

## Офлайн-инференс (без сервера)

| Файл | Описание |
|------|----------|
| **01_offline_basic.py** | Базовый пакетный инференс: список промптов → `llm.generate()`. |
| **02_offline_chat.py** | Чат-модели через `llm.chat()` с форматом сообщений system/user/assistant. |
| **03_sampling_params.py** | Разные параметры: temperature, top_p, top_k, best_of, repetition_penalty. |
| **05_batch_evaluation.py** | Пакетная обработка датасета (eval, массовая генерация). |
| **06_rag_simple.py** | Простой RAG: контекст + вопрос в одном промпте. |
| **07_offline_with_chat_template.py** | Ручное применение chat template через `transformers`. |

Запуск из корня проекта:

```bash
python examples/01_offline_basic.py
```

Или из папки `examples`:

```bash
cd examples
python 01_offline_basic.py
```

## Работа с сервером (OpenAI-совместимый API)

Сначала запустите сервер в отдельном терминале:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

| Файл | Описание |
|------|----------|
| **04_client_openai.py** | Клиент к vLLM: list models, completions, chat, стриминг. |

Запуск клиента:

```bash
python examples/04_client_openai.py
```

Переменные окружения (опционально):

- `VLLM_BASE_URL` — базовый URL API (по умолчанию `http://localhost:8000/v1`)
- `VLLM_MODEL` — имя модели на сервере

## Модели в примерах

- **facebook/opt-125m** — маленькая модель для быстрой проверки (подходит для слабого GPU/CPU).
- **Qwen/Qwen2.5-1.5B-Instruct** — лёгкая chat-модель; в примерах с чатом можно заменить на 7B/72B при наличии памяти.

В скриптах можно заменить `model_id` на путь к локальной модели, например:

```python
llm = LLM(model="/path/to/local/model")
```

## Дополнительно

- Документация vLLM: [docs.vllm.ai](https://docs.vllm.ai)
- Список поддерживаемых моделей: [Supported models](https://docs.vllm.ai/en/stable/models/supported_models/)
