# vLLM — документация для ML-инженера

**vLLM** — открытая высокопроизводительная и экономичная по памяти система инференса и сервинга больших языковых моделей (LLM).  
Изначально разработана в [Sky Computing Lab, UC Berkeley](https://sky.cs.berkeley.edu/), сейчас развивается сообществом и индустрией.

- Репозиторий: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)  
- Документация: [https://docs.vllm.ai](https://docs.vllm.ai)  
- Сайт: [https://vllm.ai](https://vllm.ai)

---

## 1. Для чего нужен vLLM

vLLM решает типичные задачи при работе с LLM в продакшене и в исследованиях:

| Задача | Описание |
|--------|----------|
| **Сервинг моделей** | Запуск модели как API (OpenAI-совместимый сервер) для чат-ботов, RAG, агентов. |
| **Пакетный инференс** | Офлайн-обработка больших списков промптов с высокой пропускной способностью. |
| **Экономия памяти** | Эффективное использование GPU за счёт PagedAttention и квантизации. |
| **Высокий throughput** | Continuous batching, chunked prefill, оптимизированные ядра (FlashAttention, FlashInfer). |
| **Гибкость** | Поддержка Hugging Face, квантизаций (GPTQ, AWQ, INT4/8, FP8), LoRA, prefix caching. |

**Кратко:** vLLM нужен, когда вы хотите **быстро и дёшево** обслуживать или прогонять LLM с максимальной производительностью на доступном железе (GPU/CPU/TPU и др.).

---

## 2. Как работает vLLM

### 2.1 Ключевые идеи

- **PagedAttention**  
  Управление ключами и значениями внимания (KV-cache) постранично, по аналогии с виртуальной памятью в ОС. Это уменьшает фрагментацию и позволяет эффективнее использовать GPU-память и увеличивать batch size.

- **Continuous batching**  
  Запросы обрабатываются непрерывным батчем: новые запросы добавляются, а завершённые — убираются по мере генерации. Это даёт высокий throughput при приемлемой задержке (в т.ч. по сравнению с статическим batching).

- **Chunked prefill**  
  Фаза prefill (обработка входного промпта) разбивается на чанки, что улучшает утилизацию и снижает пиковое потребление памяти.

- **Оптимизированные ядра**  
  Интеграция FlashAttention, FlashInfer и др. для быстрого вычисления attention на GPU.

- **Спецификация и квантизация**  
  Speculative decoding ускоряет генерацию; поддержка GPTQ, AWQ, INT4/8, FP8 уменьшает объём модели и памяти.

Подробнее: [статья про PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html), [vLLM paper (SOSP 2023)](https://arxiv.org/abs/2309.06180).

### 2.2 Два основных режима работы

1. **Offline batched inference**  
   Класс `LLM` в Python: загружаете модель один раз и вызываете `llm.generate(prompts, sampling_params)` для списка промптов. Удобно для датасетов, эвалюаций, пайплайнов обработки текста.

2. **Online serving**  
   Команда `vllm serve <model>` поднимает HTTP-сервер с OpenAI-совместимым API (`/v1/completions`, `/v1/chat/completions`, `/v1/models` и т.д.). Используется для чат-интерфейсов, RAG-сервисов, агентов и любого кода, уже завязанного на OpenAI API.

### 2.3 Поддерживаемое железо

- **GPU:** NVIDIA (CUDA), AMD (ROCm), Intel XPU  
- **CPU:** x86 (Intel/AMD), ARM (AArch64), Apple Silicon, IBM Z (s390x)  
- **Ускорители:** Google TPU, Intel Gaudi, IBM Spyre, Huawei Ascend (через плагины)

Список: [vllm.ai — Hardware](https://vllm.ai/#hardware).

---

## 3. Установка

**Требования:** Python 3.10–3.13, ОС Linux (для полной поддержки GPU). На Windows возможны ограничения (часто используют WSL2 или Docker).

### 3.1 NVIDIA CUDA (рекомендуется uv)

```bash
# Создание окружения и установка
uv venv --python 3.12 --seed
source .venv/bin/activate   # Linux/macOS
# На Windows: .venv\Scripts\activate
uv pip install vllm --torch-backend=auto
```

Или через conda:

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install --upgrade uv
uv pip install vllm --torch-backend=auto
```

Без создания venv (разовый запуск):

```bash
uv run --with vllm vllm --help
```

### 3.2 AMD ROCm

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

Нужны: Python 3.12, ROCm 7.0, glibc >= 2.35.

### 3.3 Google TPU

```bash
uv pip install vllm-tpu
```

### 3.4 Обычный pip (NVIDIA)

```bash
pip install vllm
```

Подробности по платформам и Docker: [Installation — vLLM](https://docs.vllm.ai/en/stable/getting_started/installation/).

### 3.5 Windows и ошибка `No module named 'vllm._C'`

Официальный vLLM **не собирает** нативные расширения (`vllm._C`) для Windows — только для Linux. Поэтому на Windows при `import vllm` возможна ошибка `ModuleNotFoundError: No module named 'vllm._C'`.

**Что делать:** использовать vLLM в **WSL2** (Linux) или в **Docker**; либо форк [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows) с Windows-сборками. Подробно см. **[TROUBLESHOOTING_WINDOWS.md](TROUBLESHOOTING_WINDOWS.md)** в корне проекта.

Также нужен **Python 3.10–3.13** (Python 3.9 не поддерживается).

---

## 4. Как использовать

### 4.1 Офлайн пакетный инференс (Python)

Минимальный пример:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The capital of France is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")  # или путь к локальной модели
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

Для **chat/instruct** моделей лучше применять chat template или использовать `llm.chat()`:

```python
# Вариант 1: chat-интерфейс (формат как у OpenAI)
messages_list = [
    [{"role": "user", "content": "Who won the world series in 2020?"}]
]
outputs = llm.chat(messages_list, sampling_params)

# Вариант 2: вручную через tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/chat_model")
texts = tokenizer.apply_chat_template(
    messages_list, tokenize=False, add_generation_prompt=True
)
outputs = llm.generate(texts, sampling_params)
```

По умолчанию vLLM подхватывает `generation_config.json` из репозитория модели. Чтобы использовать дефолты vLLM, при создании `LLM` укажите `generation_config="vllm"`.

### 4.2 OpenAI-совместимый сервер (онлайн-сервинг)

Запуск сервера (один модель на инстанс):

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

По умолчанию: `http://localhost:8000`. Хост/порт: `--host`, `--port`.

Примеры запросов:

```bash
# Список моделей
curl http://localhost:8000/v1/models

# Completions
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
  }'

# Chat Completions
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Who won the world series in 2020?"}
    ]
  }'
```

Через Python (OpenAI-клиент):

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# Chat
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)
print(response.choices[0].message.content)
```

Любой код, рассчитанный на OpenAI API, может переключаться на vLLM сменой `base_url` и при необходимости `api_key`.

### 4.3 Backend для Attention (опционально)

На NVIDIA можно явно задать бэкенд:

```bash
# Сервинг
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN

# Офлайн-скрипт
python script.py --attention-backend FLASHINFER
```

FlashInfer не входит в стандартные wheel’ы — его нужно ставить отдельно ([документация FlashInfer](https://docs.flashinfer.ai/)).

---

## 5. В каких случаях использовать vLLM

| Сценарий | Рекомендация |
|----------|----------------|
| **Продакшен-сервинг LLM (чат, RAG, агенты)** | Да. OpenAI-совместимый API, высокая пропускная способность, экономия памяти. |
| **Офлайн-обработка больших объёмов промптов** | Да. `LLM.generate()` с батчами, хороший throughput. |
| **Ограниченная GPU-память** | Да. PagedAttention, квантизация (AWQ, GPTQ, INT4/8, FP8), возможность подбирать batch size. |
| **Нужна замена OpenAI API локально/в своём кластере** | Да. Почти полная совместимость с форматом запросов/ответов. |
| **Модели с Hugging Face** | Да. Широкая поддержка архитектур (Llama, Qwen, Mixtral, LLaVA и др.). |
| **Мульти-LoRA, prefix caching** | Да. Встроенная поддержка. |
| **Обучение / тонкая настройка** | Частично. Есть сценарии (например, RLHF), но основной фокус vLLM — инференс и сервинг. |
| **Максимальная кастомная логика внутри модели** | Зависит от сложности. Для типовых трансформеров и MoE — отличный выбор; для экзотических архитектур — смотреть [поддерживаемые модели](https://docs.vllm.ai/en/stable/models/supported_models/). |

**Когда можно рассмотреть альтернативы:**

- Очень маленькие модели и разовые эксперименты — иногда достаточно только Hugging Face `transformers`.
- Жёсткие требования к latency на одном запросе и нестандартный стек — имеет смысл сравнивать с другими движками (TGI, TensorRT-LLM и т.д.) под ваши метрики.

---

## 6. Полезные ссылки

- [Официальная документация](https://docs.vllm.ai/en/stable/)
- [Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/)
- [Installation](https://docs.vllm.ai/en/stable/getting_started/installation/)
- [Supported models](https://docs.vllm.ai/en/stable/models/supported_models/)
- [User Guide (usage)](https://docs.vllm.ai/en/stable/usage/)
- [OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [Roadmap](https://roadmap.vllm.ai/)
- [Форум](https://vllm.ai) / [Slack](https://slack.vllm.ai) / [GitHub Issues](https://github.com/vllm-project/vllm/issues)

---

## 7. Краткая шпаргалка команд

```bash
# Установка (NVIDIA, uv)
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install vllm --torch-backend=auto

# Сервер
vllm serve <model_name_or_path> [--port 8000] [--host 0.0.0.0]

# Справка
vllm --help
vllm serve --help
```

```python
# Минимальный офлайн-инференс
from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m")
out = llm.generate(["Hello,"], SamplingParams(temperature=0.8))
print(out[0].outputs[0].text)
```

Документация составлена по состоянию официальных источников vLLM (в т.ч. [docs.vllm.ai](https://docs.vllm.ai) и [GitHub vllm-project/vllm](https://github.com/vllm-project/vllm)). Для актуальных деталей по версиям и платформам всегда сверяйтесь с официальным сайтом и репозиторием.
