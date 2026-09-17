"""LLM provider abstraction.

AnthropicProvider is the concrete, working implementation used by this
demo. MockProvider is a deterministic, key-free stand-in used by the
test suite and CI so the pipeline can be verified without secrets.

AzureOpenAIProvider is a production adapter *stub* documenting the
interface a real enterprise deployment would fill in against an Azure
OpenAI resource — swap it in for AnthropicProvider without touching the
retrieval or grounding logic.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic  # local import keeps the mock/test path dependency-free

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example).")
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class MockProvider(LLMProvider):
    """No API key required. Used by tests/CI to verify pipeline wiring
    (retrieval, grounding, citation assembly) without calling a real LLM."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "[MOCK ANSWER] Grounded on the retrieved context below.\n\n" + user_prompt[:300]


class AzureOpenAIProvider(LLMProvider):
    """Production adapter stub — see module docstring."""

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Configure an Azure OpenAI resource + deployment and implement this "
            "adapter using the `openai` SDK's AzureOpenAI client. "
            "See README 'Production deployment' section."
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


def build_llm_provider(kind: str, model: str, api_key: str | None = None) -> LLMProvider:
    if kind == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key)
    if kind == "mock":
        return MockProvider()
    if kind == "azure_openai":
        return AzureOpenAIProvider()
    raise ValueError(f"Unknown LLM provider: {kind}")
