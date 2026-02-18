"""
vLLM: офлайн-инференс с ручным применением chat template.

Полезно, когда нужно точно контролировать формат промпта
или использовать свой шаблон поверх tokenizer.

Запуск (нужна chat-модель):
  python examples/07_offline_with_chat_template.py
"""

from vllm import LLM, SamplingParams

# Для apply_chat_template
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


def main():
    messages_list = [
        [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write a one-line Python hello world."},
        ],
        [
            {"role": "user", "content": "What is 3 * 7?"},
        ],
    ]

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    if AutoTokenizer is None:
        print("Установите transformers: pip install transformers")
        return

    print("Загрузка токенайзера и модели...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id)

    # Применяем chat template вручную (без токенизации)
    prompts = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True,
    )
    # Для одной истории apply_chat_template может вернуть строку
    if isinstance(prompts, str):
        prompts = [prompts]

    sampling_params = SamplingParams(temperature=0.5, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        reply = output.outputs[0].text
        print(f"[{i+1}] Reply: {reply.strip()}")
        print()


if __name__ == "__main__":
    main()
