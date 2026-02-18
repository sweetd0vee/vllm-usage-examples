"""
vLLM: примеры разных параметров сэмплирования.

Показывает: temperature, top_p, top_k, repetition_penalty,
greedy (best_of), и различие между разными настройками.

Запуск:
  python examples/03_sampling_params.py
"""

from vllm import LLM, SamplingParams


def main():
    prompt = "The best programming language for data science is"

    model_id = "facebook/opt-125m"
    print(f"Загрузка модели {model_id}...")
    llm = LLM(model=model_id)

    # 1) Детерминированно (greedy) — один ответ, без случайности
    greedy_params = SamplingParams(temperature=0, max_tokens=15)
    out = llm.generate([prompt], greedy_params)
    print("1. Greedy (temperature=0):")
    print("   ", out[0].outputs[0].text)
    print()

    # 2) Умеренная случайность
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_tokens=15,
        repetition_penalty=1.1,
    )
    out = llm.generate([prompt], sampling_params)
    print("2. Sampling (temp=0.8, top_p=0.95, top_k=50, rep_penalty=1.1):")
    print("   ", out[0].outputs[0].text)
    print()

    # 3) Несколько вариантов для одного промпта (best_of)
    multi_params = SamplingParams(
        temperature=0.9,
        top_p=0.9,
        max_tokens=12,
        best_of=3,
    )
    outputs = llm.generate([prompt], multi_params)
    # best_of даёт несколько outputs для одного промпта
    for j, o in enumerate(outputs[0].outputs):
        print(f"3. Best-of вариант {j+1}: {o.text!r}")
    print()

    # 4) Один промпт, несколько раз — сравнение
    same_params = SamplingParams(temperature=0.7, max_tokens=10)
    repeated = llm.generate([prompt] * 3, same_params)
    print("4. Один промпт, 3 запуска (temperature=0.7):")
    for i, o in enumerate(repeated):
        print(f"   Run {i+1}: {o.outputs[0].text!r}")


if __name__ == "__main__":
    main()
