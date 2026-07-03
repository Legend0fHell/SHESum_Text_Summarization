from __future__ import annotations

import re
from dataclasses import dataclass

from .config import llm_settings


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    calls: int = 1


class BaseLLM:
    def summarize(self, prompt: str) -> LLMResult:
        raise NotImplementedError


class DryRunLLM(BaseLLM):
    def summarize(self, prompt: str) -> LLMResult:
        # Deterministic placeholder for pipeline debugging. Do not report as a paper result.
        evidence = _extract_prompt_block(prompt, "Source support") or _extract_prompt_block(prompt, "Source-derived support") or prompt
        sentences = re.split(r"(?<=[.!?。！？])\s+", evidence.strip())
        text = " ".join(sentences[:5]).strip()[:2000]
        return LLMResult(text=text, input_tokens=count_tokens(prompt), output_tokens=count_tokens(text))


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, temperature: float = 0.0):
        self.model = model
        settings = llm_settings()
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self.api_key = api_key or settings.api_key
        self.temperature = temperature

    def summarize(self, prompt: str) -> LLMResult:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        response = chat.invoke(prompt)
        text = str(response.content).strip()
        usage = getattr(response, "usage_metadata", None) or response.response_metadata.get("token_usage", {})
        return LLMResult(
            text=text,
            input_tokens=int(usage.get("input_tokens") or usage.get("prompt_tokens") or count_tokens(prompt)),
            output_tokens=int(usage.get("output_tokens") or usage.get("completion_tokens") or count_tokens(text)),
        )


def make_llm(kind: str, model: str | None = None, base_url: str | None = None, api_key: str | None = None, temperature: float | None = None) -> BaseLLM:
    if kind == "dry_run":
        return DryRunLLM()
    if kind == "openai_compatible":
        settings = llm_settings()
        resolved_model = model or settings.model
        if not resolved_model:
            raise ValueError("GRAPHSUM_LLM_MODEL or --model is required for openai_compatible LLM")
        return OpenAICompatibleLLM(
            model=resolved_model,
            base_url=base_url,
            api_key=api_key,
            temperature=settings.temperature if temperature is None else temperature,
        )
    raise ValueError(f"Unknown LLM kind: {kind}")


def count_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _extract_prompt_block(prompt: str, label: str) -> str:
    pattern = rf"{re.escape(label)}[^\n]*:\s*---\s*(.*?)\s*---"
    match = re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""
