# backend/utils/llm_client.py
"""
FinThesisGuard AI — Groq LLM Client
ArkAngel Financial Solutions

Single wrapper for all Groq LLM API calls.
Every agent imports and uses this module exclusively for inference.
Handles: async calls, JSON mode, streaming, retries, token tracking, error handling.
"""

import asyncio
import json
import time
import re
from typing import AsyncGenerator, Optional, Any
from dataclasses import dataclass, field

import groq
from groq import AsyncGroq, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

from backend.config import (
    settings,
    LLM_TEMPERATURE_FACTUAL,
    LLM_TEMPERATURE_THESIS,
    LLM_TEMPERATURE_CREATIVE,
    LLM_MAX_TOKENS_DEFAULT,
    LLM_MAX_TOKENS_THESIS,
    LLM_MAX_TOKENS_SYNTHESIS,
    LLM_RETRY_ATTEMPTS,
    LLM_RETRY_BACKOFF_BASE,
    LLM_REQUEST_TIMEOUT,
)
from backend.utils.logger import logger, log_llm_call


# ─────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────

class LLMError(Exception):
    """
    Base exception for all LLM-related failures.
    Carries structured context for debugging.
    """
    def __init__(
        self,
        message: str,
        agent_name: str = "unknown",
        attempt: int = 0,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.agent_name = agent_name
        self.attempt = attempt
        self.original_error = original_error

    def __str__(self):
        base = super().__str__()
        return (
            f"LLMError(agent={self.agent_name}, "
            f"attempt={self.attempt}): {base}"
        )


class LLMRateLimitError(LLMError):
    """Raised when Groq rate limit is exceeded."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when Groq request times out."""
    pass


class LLMJSONParseError(LLMError):
    """Raised when LLM returns invalid JSON despite JSON mode."""
    def __init__(self, message: str, raw_response: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.raw_response = raw_response


class LLMConnectionError(LLMError):
    """Raised on network/connection failures to Groq."""
    pass


# ─────────────────────────────────────────────
# RESPONSE DATACLASS
# ─────────────────────────────────────────────

@dataclass
class LLMResponse:
    """
    Structured response object from every LLM call.
    Carries content, token usage, and performance metadata.
    """
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    agent_name: str = "unknown"
    cached: bool = False
    finish_reason: str = "stop"

    @property
    def estimated_cost_usd(self) -> float:
        """Approximate cost at Groq Llama-3.3-70B pricing (~$0.59/1M tokens)."""
        return self.total_tokens * 0.00000059

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "model": self.model,
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens,
            },
            "duration_ms": self.duration_ms,
            "agent_name": self.agent_name,
            "finish_reason": self.finish_reason,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

SYSTEM_PROMPT_DEFAULT = """You are FinThesisGuard AI, an expert financial analyst assistant 
built by ArkAngel Financial Solutions. You specialize in Indian capital markets, 
banking, regulatory frameworks (SEBI, RBI), and investment thesis analysis.

Guidelines:
- Be precise and factual. Never hallucinate financial figures.
- Always cite your reasoning.
- Use Indian financial terminology correctly.
- If uncertain, say so explicitly rather than guessing.
- Structure responses clearly with proper formatting."""

SYSTEM_PROMPT_JSON = """You are FinThesisGuard AI, a financial analysis engine by ArkAngel.
You MUST respond with valid JSON only. No markdown, no explanation, no preamble.
Start your response with {{ and end with }}. 
Every field in the JSON must be properly quoted and formatted."""

SYSTEM_PROMPT_FACTUAL = """You are FinThesisGuard AI by ArkAngel Financial Solutions.
Answer only based on provided context. Be concise, precise, and factual.
Do not introduce information not present in the provided sources.
Always cite specific documents when making claims."""

SYSTEM_PROMPT_THESIS = """You are FinThesisGuard AI, an investment thesis stress-tester 
by ArkAngel Financial Solutions. Your role is to rigorously analyze investment theses, 
identify assumptions, assess risks, and find conditions that could break the thesis.
Be brutally honest. Do not sugarcoat risks. Back every claim with evidence."""


# ─────────────────────────────────────────────
# LLM CLIENT CLASS
# ─────────────────────────────────────────────

class LLMClient:
    """
    Async Groq LLM client for FinThesisGuard AI.

    Usage:
        from backend.utils.llm_client import llm_client

        # Simple call
        response = await llm_client.call_llm("What is NIM?")

        # JSON response
        data = await llm_client.call_llm_json(
            prompt="Extract assumptions from: ...",
            agent_name="agent6"
        )

        # Streaming
        async for chunk in llm_client.call_llm_stream("Explain HDFC's NIM"):
            print(chunk, end="")
    """

    def __init__(self):
        self._client: Optional[AsyncGroq] = None
        self._initialized: bool = False
        self._total_tokens_used: int = 0
        self._total_calls: int = 0
        self._failed_calls: int = 0

    def _get_client(self) -> AsyncGroq:
        """Lazy initialization of Groq async client."""
        if self._client is None:
            self._client = AsyncGroq(
                api_key=settings.groq_api_key,
                timeout=LLM_REQUEST_TIMEOUT,
                max_retries=0,                          # We handle retries manually
            )
            self._initialized = True
            logger.info(
                f"[LLM CLIENT] Groq client initialized | "
                f"model={settings.groq_model} | "
                f"timeout={LLM_REQUEST_TIMEOUT}s"
            )
        return self._client

    # ─────────────────────────────────────────
    # CORE: call_llm
    # ─────────────────────────────────────────

    async def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        agent_name: str = "unknown",
        model: Optional[str] = None,
    ) -> str:
        """
        Makes an async LLM call to Groq and returns response text.

        Args:
            prompt:        User message / instruction
            system_prompt: Override default system prompt
            temperature:   Sampling temperature (default: FACTUAL = 0.1)
            max_tokens:    Max output tokens (default: 1024)
            agent_name:    Calling agent name (for logging)
            model:         Override model (default: settings.groq_model)

        Returns:
            Response text string

        Raises:
            LLMError, LLMRateLimitError, LLMTimeoutError, LLMConnectionError
        """
        response = await self._call_with_retry(
            prompt=prompt,
            system_prompt=system_prompt or SYSTEM_PROMPT_DEFAULT,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE_FACTUAL,
            max_tokens=max_tokens or LLM_MAX_TOKENS_DEFAULT,
            agent_name=agent_name,
            model=model or settings.groq_model,
            json_mode=False,
        )
        return response.content

    # ─────────────────────────────────────────
    # CORE: call_llm_json
    # ─────────────────────────────────────────

    async def call_llm_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        agent_name: str = "unknown",
        model: Optional[str] = None,
        fallback_on_parse_error: bool = True,
    ) -> dict:
        """
        Makes an async LLM call and forces structured JSON output.
        Parses and returns the response as a Python dict.

        Args:
            prompt:                 User message / instruction
            system_prompt:          Override (defaults to JSON system prompt)
            temperature:            Sampling temperature (default: 0.1)
            max_tokens:             Max output tokens
            agent_name:             Calling agent for logging
            model:                  Override model
            fallback_on_parse_error: If True, attempt JSON repair before raising

        Returns:
            Parsed dict from LLM response

        Raises:
            LLMJSONParseError if JSON cannot be parsed even after repair
        """
        response = await self._call_with_retry(
            prompt=prompt,
            system_prompt=system_prompt or SYSTEM_PROMPT_JSON,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE_FACTUAL,
            max_tokens=max_tokens or LLM_MAX_TOKENS_DEFAULT,
            agent_name=agent_name,
            model=model or settings.groq_model,
            json_mode=True,
        )

        raw = response.content.strip()

        # Attempt 1: Direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Attempt 2: Extract JSON block from markdown fences
        if fallback_on_parse_error:
            extracted = _extract_json_from_markdown(raw)
            if extracted:
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    pass

            # Attempt 3: Repair common LLM JSON errors
            repaired = _repair_json(raw)
            if repaired:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

        logger.error(
            f"[LLM JSON PARSE FAILED] agent={agent_name} | "
            f"raw_preview={raw[:200]}"
        )
        raise LLMJSONParseError(
            message=f"Failed to parse JSON response from LLM",
            raw_response=raw,
            agent_name=agent_name,
        )

    # ─────────────────────────────────────────
    # CORE: call_llm_stream
    # ─────────────────────────────────────────

    async def call_llm_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        agent_name: str = "unknown",
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Makes a streaming LLM call and yields text chunks as they arrive.

        Args:
            prompt:        User message
            system_prompt: Override system prompt
            temperature:   Sampling temperature
            max_tokens:    Max output tokens
            agent_name:    Calling agent for logging
            model:         Override model

        Yields:
            String chunks of response text

        Usage:
            async for chunk in llm_client.call_llm_stream("Explain..."):
                yield chunk  # Pass to FastAPI StreamingResponse
        """
        client = self._get_client()
        _temperature = temperature if temperature is not None else LLM_TEMPERATURE_FACTUAL
        _model = model or settings.groq_model
        _system = system_prompt or SYSTEM_PROMPT_DEFAULT
        _max_tokens = max_tokens or LLM_MAX_TOKENS_SYNTHESIS

        messages = [
            {"role": "system", "content": _system},
            {"role": "user", "content": prompt},
        ]

        start_time = time.perf_counter()
        total_chars = 0

        logger.debug(
            f"[LLM STREAM START] agent={agent_name} | model={_model} | "
            f"prompt_len={len(prompt)}"
        )

        try:
            stream = await client.chat.completions.create(
                model=_model,
                messages=messages,
                temperature=_temperature,
                max_tokens=_max_tokens,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    total_chars += len(delta)
                    yield delta

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"[LLM STREAM END] agent={agent_name} | "
                f"chars={total_chars} | {duration_ms:.0f}ms"
            )
            self._total_calls += 1

        except RateLimitError as e:
            logger.warning(f"[LLM STREAM RATE LIMIT] agent={agent_name}: {e}")
            raise LLMRateLimitError(
                message=f"Groq rate limit exceeded during streaming",
                agent_name=agent_name,
                original_error=e,
            )
        except APITimeoutError as e:
            logger.error(f"[LLM STREAM TIMEOUT] agent={agent_name}: {e}")
            raise LLMTimeoutError(
                message=f"Groq request timed out during streaming",
                agent_name=agent_name,
                original_error=e,
            )
        except Exception as e:
            logger.exception(f"[LLM STREAM ERROR] agent={agent_name}: {e}")
            raise LLMError(
                message=f"Streaming failed: {str(e)}",
                agent_name=agent_name,
                original_error=e,
            )

    # ─────────────────────────────────────────
    # CONVENIENCE WRAPPERS
    # ─────────────────────────────────────────

    async def call_factual(
        self,
        prompt: str,
        agent_name: str = "unknown",
        max_tokens: int = LLM_MAX_TOKENS_DEFAULT,
    ) -> str:
        """
        Low temperature (0.1) call for factual/RAG queries.
        Uses SYSTEM_PROMPT_FACTUAL.
        """
        return await self.call_llm(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_FACTUAL,
            temperature=LLM_TEMPERATURE_FACTUAL,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )

    async def call_thesis(
        self,
        prompt: str,
        agent_name: str = "unknown",
        max_tokens: int = LLM_MAX_TOKENS_THESIS,
    ) -> str:
        """
        Medium temperature (0.3) call for thesis analysis.
        Uses SYSTEM_PROMPT_THESIS.
        """
        return await self.call_llm(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_THESIS,
            temperature=LLM_TEMPERATURE_THESIS,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )

    async def call_thesis_json(
        self,
        prompt: str,
        agent_name: str = "unknown",
        max_tokens: int = LLM_MAX_TOKENS_THESIS,
    ) -> dict:
        """
        Medium temperature JSON call for structured thesis output.
        """
        return await self.call_llm_json(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_JSON,
            temperature=LLM_TEMPERATURE_THESIS,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )

    async def call_fast(
        self,
        prompt: str,
        agent_name: str = "unknown",
        max_tokens: int = 512,
    ) -> str:
        """
        Uses the fast model (llama-3.1-8b-instant) for simple/quick tasks.
        Agent 1 (acronym resolver) and health checks use this.
        """
        return await self.call_llm(
            prompt=prompt,
            temperature=LLM_TEMPERATURE_FACTUAL,
            max_tokens=max_tokens,
            agent_name=agent_name,
            model=settings.groq_model_fast,
        )

    async def call_fast_json(
        self,
        prompt: str,
        agent_name: str = "unknown",
        max_tokens: int = 512,
    ) -> dict:
        """
        Fast model JSON call for lightweight structured tasks.
        """
        return await self.call_llm_json(
            prompt=prompt,
            temperature=LLM_TEMPERATURE_FACTUAL,
            max_tokens=max_tokens,
            agent_name=agent_name,
            model=settings.groq_model_fast,
        )

    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns usage statistics for monitoring."""
        return {
            "total_calls": self._total_calls,
            "failed_calls": self._failed_calls,
            "total_tokens_used": self._total_tokens_used,
            "success_rate": (
                round((self._total_calls - self._failed_calls) / self._total_calls * 100, 1)
                if self._total_calls > 0 else 0.0
            ),
            "estimated_total_cost_usd": round(
                self._total_tokens_used * 0.00000059, 4
            ),
        }

    async def health_check(self) -> dict:
        """
        Validates Groq connectivity with a minimal test call.
        Used by /health and /ready endpoints.
        """
        start = time.perf_counter()
        try:
            response = await self.call_fast(
                prompt="Reply with the single word: OK",
                agent_name="health_check",
                max_tokens=5,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            return {
                "status": "ok",
                "response": response.strip(),
                "model": settings.groq_model_fast,
                "latency_ms": round(duration_ms, 1),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": settings.groq_model_fast,
            }

    # ─────────────────────────────────────────
    # INTERNAL: RETRY LOGIC
    # ─────────────────────────────────────────

    async def _call_with_retry(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        agent_name: str,
        model: str,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Core method with retry + exponential backoff.
        Handles all Groq API error types with appropriate responses.

        Retry strategy:
            Attempt 1: Immediate
            Attempt 2: Wait 2s
            Attempt 3: Wait 4s
            After 3 failures: Raise LLMError
        """
        last_exception: Optional[Exception] = None
        start_time = time.perf_counter()

        for attempt in range(1, LLM_RETRY_ATTEMPTS + 1):
            try:
                logger.debug(
                    f"[LLM CALL] agent={agent_name} | model={model} | "
                    f"attempt={attempt}/{LLM_RETRY_ATTEMPTS} | "
                    f"temp={temperature} | max_tokens={max_tokens} | "
                    f"json={json_mode} | prompt_len={len(prompt)}"
                )

                response = await self._execute_call(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                    json_mode=json_mode,
                )

                duration_ms = (time.perf_counter() - start_time) * 1000
                response.duration_ms = duration_ms
                response.agent_name = agent_name

                # Update stats
                self._total_calls += 1
                self._total_tokens_used += response.total_tokens

                # Log the call
                log_llm_call(
                    agent_name=agent_name,
                    model=model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    duration_ms=duration_ms,
                    success=True,
                )

                return response

            # ── Rate Limit: wait longer before retry ──────────────────────────
            except RateLimitError as e:
                last_exception = e
                wait = LLM_RETRY_BACKOFF_BASE ** attempt * 2    # Longer wait for rate limits
                logger.warning(
                    f"[LLM RATE LIMIT] agent={agent_name} | attempt={attempt} | "
                    f"waiting {wait}s before retry"
                )
                if attempt < LLM_RETRY_ATTEMPTS:
                    await asyncio.sleep(wait)
                else:
                    self._failed_calls += 1
                    log_llm_call(agent_name, model, 0, 0, 0, False, str(e))
                    raise LLMRateLimitError(
                        message=f"Groq rate limit exceeded after {LLM_RETRY_ATTEMPTS} attempts",
                        agent_name=agent_name,
                        attempt=attempt,
                        original_error=e,
                    )

            # ── Timeout: retry immediately ─────────────────────────────────────
            except APITimeoutError as e:
                last_exception = e
                wait = LLM_RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.warning(
                    f"[LLM TIMEOUT] agent={agent_name} | attempt={attempt} | "
                    f"waiting {wait}s before retry"
                )
                if attempt < LLM_RETRY_ATTEMPTS:
                    await asyncio.sleep(wait)
                else:
                    self._failed_calls += 1
                    log_llm_call(agent_name, model, 0, 0, 0, False, str(e))
                    raise LLMTimeoutError(
                        message=f"Groq request timed out after {LLM_RETRY_ATTEMPTS} attempts",
                        agent_name=agent_name,
                        attempt=attempt,
                        original_error=e,
                    )

            # ── Connection Error: retry with backoff ───────────────────────────
            except APIConnectionError as e:
                last_exception = e
                wait = LLM_RETRY_BACKOFF_BASE ** attempt
                logger.error(
                    f"[LLM CONNECTION ERROR] agent={agent_name} | attempt={attempt} | "
                    f"error={e} | waiting {wait}s"
                )
                if attempt < LLM_RETRY_ATTEMPTS:
                    await asyncio.sleep(wait)
                else:
                    self._failed_calls += 1
                    log_llm_call(agent_name, model, 0, 0, 0, False, str(e))
                    raise LLMConnectionError(
                        message=f"Cannot connect to Groq API: {str(e)}",
                        agent_name=agent_name,
                        attempt=attempt,
                        original_error=e,
                    )

            # ── API Status Error (4xx/5xx) ─────────────────────────────────────
            except APIStatusError as e:
                last_exception = e
                # 4xx errors (bad request, auth) — don't retry
                if 400 <= e.status_code < 500:
                    self._failed_calls += 1
                    log_llm_call(agent_name, model, 0, 0, 0, False, str(e))
                    raise LLMError(
                        message=f"Groq API error {e.status_code}: {e.message}",
                        agent_name=agent_name,
                        attempt=attempt,
                        original_error=e,
                    )
                # 5xx errors — retry
                wait = LLM_RETRY_BACKOFF_BASE ** attempt
                logger.error(
                    f"[LLM SERVER ERROR] agent={agent_name} | "
                    f"status={e.status_code} | waiting {wait}s"
                )
                if attempt < LLM_RETRY_ATTEMPTS:
                    await asyncio.sleep(wait)
                else:
                    self._failed_calls += 1
                    log_llm_call(agent_name, model, 0, 0, 0, False, str(e))
                    raise LLMError(
                        message=f"Groq server error {e.status_code} after {LLM_RETRY_ATTEMPTS} attempts",
                        agent_name=agent_name,
                        attempt=attempt,
                        original_error=e,
                    )

            # ── Unknown Errors — don't retry ──────────────────────────────────
            except Exception as e:
                self._failed_calls += 1
                log_llm_call(agent_name, model, 0, 0, 0, False, str(e))
                logger.exception(
                    f"[LLM UNKNOWN ERROR] agent={agent_name} | error={type(e).__name__}: {e}"
                )
                raise LLMError(
                    message=f"Unexpected LLM error: {str(e)}",
                    agent_name=agent_name,
                    attempt=attempt,
                    original_error=e,
                )

        # Should never reach here, but satisfy type checker
        self._failed_calls += 1
        raise LLMError(
            message=f"All {LLM_RETRY_ATTEMPTS} LLM retry attempts failed",
            agent_name=agent_name,
            original_error=last_exception,
        )

    async def _execute_call(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        model: str,
        json_mode: bool,
    ) -> LLMResponse:
        """
        Executes a single Groq API call (no retry logic here).
        Extracts token counts and wraps in LLMResponse.
        """
        client = self._get_client()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Enable JSON mode when requested
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        call_start = time.perf_counter()
        completion = await client.chat.completions.create(**kwargs)
        call_duration = (time.perf_counter() - call_start) * 1000

        content = completion.choices[0].message.content or ""
        finish_reason = completion.choices[0].finish_reason or "stop"

        # Extract token usage (Groq always returns usage)
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        # Warn if response was cut short
        if finish_reason == "length":
            logger.warning(
                f"[LLM TRUNCATED] model={model} | max_tokens={max_tokens} hit | "
                f"consider increasing max_tokens"
            )

        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=call_duration,
            finish_reason=finish_reason,
        )


# ─────────────────────────────────────────────
# JSON REPAIR UTILITIES
# ─────────────────────────────────────────────

def _extract_json_from_markdown(text: str) -> Optional[str]:
    """
    Extracts JSON content from markdown code fences.
    LLMs sometimes wrap JSON in ```json ... ``` blocks.

    Examples:
        ```json\n{"key": "value"}\n``` → '{"key": "value"}'
        ```\n{"key": "value"}\n```    → '{"key": "value"}'
    """
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"`([\s\S]*?)`",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if candidate.startswith("{") or candidate.startswith("["):
                return candidate
    return None


def _repair_json(text: str) -> Optional[str]:
    """
    Attempts to repair common LLM JSON generation errors.

    Fixes:
    - Trailing commas before } or ]
    - Single quotes instead of double quotes
    - Unquoted keys
    - Truncated JSON (adds closing braces)
    - Python True/False/None instead of JSON true/false/null
    """
    if not text:
        return None

    # Extract just the JSON portion if there's surrounding text
    start_brace = text.find("{")
    start_bracket = text.find("[")

    if start_brace == -1 and start_bracket == -1:
        return None

    if start_brace == -1:
        start = start_bracket
    elif start_bracket == -1:
        start = start_brace
    else:
        start = min(start_brace, start_bracket)

    text = text[start:]

    # Fix Python literals → JSON literals
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)

    # Fix trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # Fix single-quoted strings → double-quoted
    text = re.sub(r"(?<![\\])'", '"', text)

    # Attempt to close unclosed JSON by counting braces
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    text += '}' * max(0, open_braces)
    text += ']' * max(0, open_brackets)

    # Validate it's now parseable
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────
# PROMPT BUILDER UTILITIES
# ─────────────────────────────────────────────

def build_rag_prompt(
    query: str,
    context_chunks: list[dict],
    max_context_chars: int = 8000,
) -> str:
    """
    Builds a RAG prompt by injecting retrieved context chunks.
    Truncates context to fit within token budget.

    Args:
        query:             The user's financial question
        context_chunks:    List of retrieved doc chunks with metadata
        max_context_chars: Max characters of context to inject

    Returns:
        Complete prompt string ready for LLM
    """
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("metadata", {})
        source_label = (
            f"[Source {i}] {source.get('title', 'Unknown')} "
            f"({source.get('source_type', 'unknown').upper()}, "
            f"{source.get('date', 'date unknown')})"
        )
        chunk_text = chunk.get("text", "")

        entry = f"{source_label}\n{chunk_text}\n"
        if total_chars + len(entry) > max_context_chars:
            break

        context_parts.append(entry)
        total_chars += len(entry)

    context_str = "\n---\n".join(context_parts)

    return f"""Answer the following financial question using ONLY the provided sources below.
Always cite sources using [Source N] notation inline.
If the answer cannot be found in the sources, say "Insufficient data in available sources."
Never hallucinate figures. State conflicts between sources explicitly.

QUESTION: {query}

SOURCES:
{context_str}

ANSWER:"""


def build_json_extraction_prompt(
    instruction: str,
    input_text: str,
    output_schema: str,
) -> str:
    """
    Builds a structured JSON extraction prompt.

    Args:
        instruction:   What to extract
        input_text:    Text to extract from
        output_schema: JSON schema description for the output

    Returns:
        Complete prompt string
    """
    return f"""{instruction}

INPUT TEXT:
{input_text}

OUTPUT REQUIREMENTS:
Return a JSON object matching this schema exactly:
{output_schema}

Respond with valid JSON only. No explanation. No markdown.
JSON:"""


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# All agents import this directly
# ─────────────────────────────────────────────

llm_client = LLMClient()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    # Main client
    "llm_client",
    "LLMClient",

    # Response type
    "LLMResponse",

    # Exceptions
    "LLMError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMJSONParseError",
    "LLMConnectionError",

    # System prompts
    "SYSTEM_PROMPT_DEFAULT",
    "SYSTEM_PROMPT_JSON",
    "SYSTEM_PROMPT_FACTUAL",
    "SYSTEM_PROMPT_THESIS",

    # Prompt builders
    "build_rag_prompt",
    "build_json_extraction_prompt",
]
