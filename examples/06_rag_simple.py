"""
vLLM: простой пример RAG-стиля (контекст + вопрос).

Модели подаётся контекст (например, из векторного поиска) и вопрос;
ответ должна давать на основе контекста.

Запуск:
  python examples/06_rag_simple.py
"""

from vllm import LLM, SamplingParams


def build_rag_prompt(context: str, question: str) -> str:
    """Собираем промпт в формате контекст + вопрос."""
    return f"""Based on the following context, answer the question. Be brief.

Context:
{context}

Question: {question}

Answer:"""


def main():
    # Имитация контекста из поиска по документам
    context = """
    vLLM is a fast and easy-to-use library for LLM inference and serving.
    It uses PagedAttention for efficient memory management and supports
    continuous batching for high throughput. vLLM is compatible with
    OpenAI API and supports many Hugging Face models.
    """

    questions = [
        "What is vLLM?",
        "What does vLLM use for memory management?",
        "Is vLLM compatible with OpenAI API?",
    ]

    sampling_params = SamplingParams(temperature=0.2, max_tokens=80)

    model_id = "facebook/opt-125m"
    print(f"Загрузка модели {model_id}...")
    llm = LLM(model=model_id)

    # Формируем промпты RAG
    prompts = [build_rag_prompt(context, q) for q in questions]
    outputs = llm.generate(prompts, sampling_params)

    for q, out in zip(questions, outputs):
        answer = out.outputs[0].text.strip()
        print(f"Q: {q}")
        print(f"A: {answer}")
        print()


if __name__ == "__main__":
    main()
