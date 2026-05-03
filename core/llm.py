from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from core.config import api_keys, models, paths

logger = logging.getLogger(__name__)

preferred_provider: str = "auto"


@dataclass
class LLMResponse:
    text: str
    model_used: str
    fallback_level: int
    error: str | None = None


def _load_demo_cache() -> dict[str, str]:
    cache_file = paths.DEMO_CACHE / "responses.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _cache_key(prompt: str) -> str:
    import hashlib
    return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()


def _check_cache(prompt: str) -> str | None:
    cache = _load_demo_cache()
    key = _cache_key(prompt)
    if key in cache:
        return cache[key]
    legacy_key = prompt[:200].strip().lower()
    for cached_key, cached_response in cache.items():
        if cached_key.strip().lower() == legacy_key:
            return cached_response
    return None


def _call_gemini(prompt: str, system_prompt: str | None = None) -> str:
    from google import genai

    client = genai.Client(api_key=api_keys.GEMINI)

    config = genai.types.GenerateContentConfig(
        temperature=models.LLM_TEMPERATURE,
        max_output_tokens=models.LLM_MAX_TOKENS,
    )
    if system_prompt:
        config.system_instruction = system_prompt

    response = client.models.generate_content(
        model=models.GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    return response.text


def _call_groq(prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
    from groq import Groq

    client = Groq(api_key=api_keys.GROQ)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model or models.GROQ_MODEL_PRIMARY,
        messages=messages,
        temperature=models.LLM_TEMPERATURE,
        max_tokens=models.LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content


def call_llm(prompt: str, system_prompt: str | None = None) -> LLMResponse:
    errors: list[str] = []
    pref = preferred_provider

    if pref in ("auto", "gemini") and api_keys.has_gemini():
        try:
            text = _call_gemini(prompt, system_prompt)
            if text and text.strip():
                return LLMResponse(
                    text=text.strip(),
                    model_used=models.GEMINI_MODEL,
                    fallback_level=0,
                )
        except Exception as e:
            errors.append(f"Gemini: {e}")
            logger.warning("Gemini failed: %s", e)

    if pref in ("auto", "groq_primary") and api_keys.has_groq():
        try:
            text = _call_groq(prompt, system_prompt, models.GROQ_MODEL_PRIMARY)
            if text and text.strip():
                return LLMResponse(
                    text=text.strip(),
                    model_used=models.GROQ_MODEL_PRIMARY,
                    fallback_level=1,
                )
        except Exception as e:
            errors.append(f"Groq 70B: {e}")
            logger.warning("Groq 70B failed: %s", e)

    if pref in ("auto", "groq_fallback") and api_keys.has_groq():
        try:
            text = _call_groq(prompt, system_prompt, models.GROQ_MODEL_SECONDARY)
            if text and text.strip():
                return LLMResponse(
                    text=text.strip(),
                    model_used=models.GROQ_MODEL_SECONDARY,
                    fallback_level=2,
                )
        except Exception as e:
            errors.append(f"Groq 8B: {e}")
            logger.warning("Groq 8B failed: %s", e)

    if pref == "auto" and api_keys.has_groq():
        try:
            text = _call_groq(prompt, system_prompt, models.GROQ_MODEL_TERTIARY)
            if text and text.strip():
                return LLMResponse(
                    text=text.strip(),
                    model_used=models.GROQ_MODEL_TERTIARY,
                    fallback_level=3,
                )
        except Exception as e:
            errors.append(f"Groq GPT-OSS-20B: {e}")
            logger.warning("Groq GPT-OSS-20B failed: %s", e)

    if pref == "auto" and api_keys.has_groq():
        try:
            text = _call_groq(prompt, system_prompt, models.GROQ_MODEL_QUATERNARY)
            if text and text.strip():
                return LLMResponse(
                    text=text.strip(),
                    model_used=models.GROQ_MODEL_QUATERNARY,
                    fallback_level=4,
                )
        except Exception as e:
            errors.append(f"Groq Scout: {e}")
            logger.warning("Groq Scout failed: %s", e)

    cached = _check_cache(prompt)
    if cached:
        return LLMResponse(
            text=cached,
            model_used="demo_cache",
            fallback_level=5,
        )

    error_summary = " | ".join(errors) if errors else "No API keys configured"
    return LLMResponse(
        text="",
        model_used="none",
        fallback_level=-1,
        error=f"All LLM backends failed: {error_summary}",
    )


def save_to_demo_cache(prompt: str, response: str) -> None:
    cache_file = paths.DEMO_CACHE / "responses.json"
    cache = _load_demo_cache()
    key = _cache_key(prompt)
    cache[key] = response
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
