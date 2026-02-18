"""
vLLM: офлайн-инференс для chat/instruct моделей.

Используется llm.chat() с форматом сообщений как в OpenAI API.
Либо ручное применение chat template через tokenizer.

Запуск (нужна chat-модель и достаточно GPU памяти):
  python examples/02_offline_chat.py

Пример модели: Qwen/Qwen2.5-1.5B-Instruct (лёгкая) или Qwen/Qwen2.5-7B-Instruct.
"""

from vllm import LLM, SamplingParams


def main():
    # Сообщения в формате OpenAI: system, user, assistant
    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."},
        ],
        [
            {"role": "user", "content": "Write a very short haiku about coding."},
        ],
    ]

    sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

    # Подставьте свою chat-модель
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Загрузка модели {model_id}...")
    llm = LLM(model=model_id)

    # Чат-интерфейс: сам применяет chat template
    print("Генерация через llm.chat()...")
    outputs = llm.chat(messages_list, sampling_params)

    for i, output in enumerate(outputs):
        reply = output.outputs[0].text
        last_user = next(
            (m["content"] for m in reversed(messages_list[i]) if m["role"] == "user"),
            "",
        )
        print(f"[{i+1}] User: {last_user[:60]}...")
        print(f"     Assistant: {reply.strip()}")
        print()


if __name__ == "__main__":
    main()
