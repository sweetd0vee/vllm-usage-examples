"""
vLLM: клиент к OpenAI-совместимому серверу vLLM.

Сначала запустите сервер в отдельном терминале:
  vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000

Затем запустите этот скрипт:
  python examples/04_client_openai.py

Требуется: pip install openai
"""

import os
from openai import OpenAI


# Адрес vLLM сервера
BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def completions_example():
    """Completions API — один промпт, продолжение текста."""
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL)

    completion = client.completions.create(
        model=MODEL,
        prompt="San Francisco is a",
        max_tokens=20,
        temperature=0,
    )
    text = completion.choices[0].text
    print("Completions:")
    print("  Prompt: 'San Francisco is a'")
    print("  Result:", text)
    print()


def chat_example():
    """Chat Completions API — диалог в формате system/user/assistant."""
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
            {"role": "user", "content": "What is 2 + 2? One short sentence."},
        ],
        max_tokens=50,
        temperature=0,
    )
    reply = response.choices[0].message.content
    print("Chat Completions:")
    print("  User: What is 2 + 2?")
    print("  Assistant:", reply)
    print()


def chat_streaming_example():
    """Стриминг ответа по токенам (если сервер поддерживает)."""
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        max_tokens=50,
        stream=True,
    )
    print("Chat Streaming:")
    print("  ", end="")
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def list_models():
    """Список моделей на сервере."""
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL)
    models = client.models.list()
    print("Доступные модели:")
    for m in models.data:
        print(" ", m.id)
    print()


def main():
    print(f"Подключение к {BASE_URL}, модель: {MODEL}\n")
    try:
        list_models()
        completions_example()
        chat_example()
        chat_streaming_example()
    except Exception as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что сервер запущен: vllm serve <model> --port 8000")


if __name__ == "__main__":
    main()
