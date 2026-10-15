"""Factory for the chat model. Swap providers with the LLM_PROVIDER env variable."""

import os

from src import config


def get_llm(provider: str | None = None):
    provider = (provider or config.LLM_PROVIDER).lower()

    if provider == "none":
        return None

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.TEMPERATURE,
        )

    if provider == "groq":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY in your .env file to use Groq.")
        # Groq exposes an OpenAI-compatible API.
        return ChatOpenAI(
            model=config.GROQ_MODEL,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=config.TEMPERATURE,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY in your .env file to use OpenAI.")
        return ChatOpenAI(model=config.OPENAI_MODEL, temperature=config.TEMPERATURE)

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use ollama, groq, openai or none.")


def describe(provider: str | None = None) -> str:
    provider = (provider or config.LLM_PROVIDER).lower()
    model = {
        "ollama": config.OLLAMA_MODEL,
        "groq": config.GROQ_MODEL,
        "openai": config.OPENAI_MODEL,
    }.get(provider)
    return f"{provider} / {model}" if model else "none (retrieval only)"
