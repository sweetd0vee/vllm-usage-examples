"""
vLLM: пакетная обработка списка промптов (офлайн).

Полезно для:
  - оценки модели на датасете (eval)
  - массовой генерации (синтетические данные, аугментация)
  - обработки очереди задач одним батчем

Запуск:
  python examples/05_batch_evaluation.py
"""

from vllm import LLM, SamplingParams


def main():
    # Имитация датасета: список вопросов или промптов
    prompts = [
        "What is the capital of France? Answer:",
        "What is 15 + 27? Answer:",
        "Name one programming language. Answer:",
        "The opposite of hot is",
        "Python is a programming language. True or false?",
    ]

    sampling_params = SamplingParams(
        temperature=0.3,   # низкая температура для более предсказуемых ответов
        max_tokens=32,
        stop=["\n", "Question:"],  # остановка на новой строке или метке
    )

    model_id = "facebook/opt-125m"
    print(f"Загрузка модели {model_id}...")
    llm = LLM(model=model_id)

    # Один вызов — обрабатывается весь батч
    print(f"Обработка {len(prompts)} промптов...")
    outputs = llm.generate(prompts, sampling_params)

    # Сбор результатов (например, для записи в JSON/CSV)
    results = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        answer = output.outputs[0].text.strip()
        results.append({"prompt": prompt, "answer": answer})
        print(f"[{i+1}] Q: {prompt[:50]}...")
        print(f"    A: {answer}")
        print()

    # Дальше можно: сохранить results, вычислить метрики и т.д.
    # import json
    # with open("eval_results.json", "w") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Всего обработано: {len(results)} запросов.")


if __name__ == "__main__":
    main()
