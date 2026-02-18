"""
vLLM: базовый офлайн-инференс (batch inference).

Запуск: нужна установленная модель и GPU/CPU.
  python examples/01_offline_basic.py

Модель по умолчанию: facebook/opt-125m (маленькая, для проверки).
Замените на свою (например Qwen/Qwen2.5-1.5B-Instruct) при наличии памяти.
"""

from vllm import LLM, SamplingParams


def main():
    # Промпты для генерации
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Параметры сэмплирования
    sampling_params = SamplingParams(
        temperature=0.8,   # случайность (0 = детерминированно)
        top_p=0.95,       # nucleus sampling
        max_tokens=32,
    )

    # Загрузка модели (скачается с Hugging Face при первом запуске)
    # Для локальной модели: model="/path/to/model"
    model_id = "facebook/opt-125m"
    print(f"Загрузка модели {model_id}...")
    llm = LLM(model=model_id)

    # Пакетная генерация
    print("Генерация...")
    outputs = llm.generate(prompts, sampling_params)

    # Вывод результатов
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"[{i+1}] Prompt: {prompt!r}")
        print(f"     Generated: {generated!r}")
        print()


if __name__ == "__main__":
    main()
