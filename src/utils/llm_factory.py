"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import Field

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class GeminiFailoverChatModel(BaseChatModel):
    """Retry a Gemini generation with the next project key after RPD exhaustion."""

    models: list[Any] = Field(exclude=True)
    active_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "gemini-key-failover"

    @staticmethod
    def _is_daily_quota_error(error: Exception) -> bool:
        message = str(error).lower()
        return "resource_exhausted" in message and "perday" in message

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        for index in range(self.active_index, len(self.models)):
            try:
                result = self.models[index]._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                self.active_index = index
                return result
            except Exception as error:
                if not self._is_daily_quota_error(error) or index == len(self.models) - 1:
                    raise
                print(f"⚠️  Gemini key {index + 1} hết quota ngày, chuyển sang key {index + 2}...")
        raise RuntimeError("Không còn Gemini API key khả dụng")


class GeminiQuotaSafeEmbeddings(Embeddings):
    """Wrap Gemini embeddings to stay below the free-tier per-minute quota.

    Gemini counts each embedded text toward the quota, including vectorstore
    queries.  The wrapper keeps a small buffer below 100 texts/minute and
    retries once after the API-provided quota window has elapsed.
    """

    def __init__(self, embeddings: Embeddings, max_texts_per_minute: int = 95):
        self._embeddings = embeddings
        self._max_texts_per_minute = max_texts_per_minute
        self._window_started = time.monotonic()
        self._texts_used = 0

    @staticmethod
    def _is_quota_error(error: Exception) -> bool:
        message = str(error)
        return "RESOURCE_EXHAUSTED" in message or "429" in message

    def _reset_window_if_needed(self) -> None:
        if time.monotonic() - self._window_started >= 60:
            self._window_started = time.monotonic()
            self._texts_used = 0

    def _wait_for_next_window(self) -> None:
        delay = max(1, 61 - (time.monotonic() - self._window_started))
        print(f"⏳ Gemini embedding quota: chờ {delay:.0f}s để tiếp tục...")
        time.sleep(delay)
        self._window_started = time.monotonic()
        self._texts_used = 0

    def _run_with_quota(self, operation, text_count: int):
        self._reset_window_if_needed()
        if self._texts_used + text_count > self._max_texts_per_minute:
            self._wait_for_next_window()
        self._texts_used += text_count

        try:
            return operation()
        except Exception as error:
            if not self._is_quota_error(error):
                raise
            self._wait_for_next_window()
            self._texts_used = text_count
            return operation()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        start = 0
        while start < len(texts):
            self._reset_window_if_needed()
            remaining = self._max_texts_per_minute - self._texts_used
            if remaining <= 0:
                self._wait_for_next_window()
                remaining = self._max_texts_per_minute
            batch = texts[start : start + min(remaining, 100)]
            vectors.extend(
                self._run_with_quota(
                    lambda: self._embeddings.embed_documents(batch), len(batch)
                )
            )
            start += len(batch)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._run_with_quota(
            lambda: self._embeddings.embed_query(text), 1
        )

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def get_llm(
    provider: str = None,
    temperature: float = 0.0,
    model_override: str = None,
):
    """
    Trả về BaseChatModel tương ứng với provider được chọn.

    Args:
        provider    : "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                      Mặc định: đọc PROVIDER từ .env (config.PROVIDER)
        temperature : độ ngẫu nhiên (0.0 = tất định, 1.0 = sáng tạo)
        model_override: model Gemini dùng tạm cho một tác vụ chuyên biệt.

    Returns:
        BaseChatModel instance sẵn sàng sử dụng

    Raises:
        ValueError nếu provider không hợp lệ
        ImportError nếu package tương ứng chưa được cài đặt
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Gemini 2.0 Flash has been retired.  Preserve compatibility with
        # existing .env files while using the model recommended by the API.
        model = model_override or config.GEMINI_MODEL
        if model in {"gemini-2.0-flash", "gemini-2.0-flash-lite"}:
            model = "gemini-3.6-flash"

        # Keep a margin below the free-tier 15 RPM limit.  The evaluator uses
        # full Flash and can safely run a little closer to that limit.
        requests_per_second = 0.2 if "flash-lite" in model else 0.24
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
        keys = config.GOOGLE_API_KEYS or [config.GOOGLE_API_KEY]
        models = [
            ChatGoogleGenerativeAI(
                model=model,
                google_api_key=key,
                temperature=temperature,
                rate_limiter=InMemoryRateLimiter(
                    requests_per_second=requests_per_second,
                    check_every_n_seconds=0.1,
                    max_bucket_size=1,
                ),
            )
            for key in keys
        ]
        return GeminiFailoverChatModel(models=models, rate_limiter=rate_limiter)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    elif provider == "openrouter":
        # OpenRouter dùng OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance tương ứng với provider được chọn.

    Lưu ý quan trọng:
        - Anthropic KHÔNG có Embeddings API → tự động fallback về OpenAI embeddings
        - OpenRouter cũng dùng OpenAI embeddings (không có API embeddings riêng)
        - Ollama cần model embedding riêng (mặc định: nomic-embed-text)
          Cài đặt: ollama pull nomic-embed-text

    Args:
        provider: "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                  Mặc định: đọc PROVIDER từ .env

    Returns:
        Embeddings instance sẵn sàng sử dụng
    """
    provider = (provider or config.PROVIDER).lower()

    if provider in ("openai", "openrouter"):
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # ``models/embedding-001`` was the legacy Gemini embedding endpoint.
        # Keep old .env files usable while routing them to the current model.
        embedding_model = config.GEMINI_EMBEDDING_MODEL
        if embedding_model in {
            "embedding-001",
            "models/embedding-001",
            "text-embedding-004",
            "models/text-embedding-004",
        }:
            embedding_model = "gemini-embedding-001"
        return GeminiQuotaSafeEmbeddings(
            GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=config.GOOGLE_API_KEY,
            )
        )

    elif provider == "anthropic":
        # Anthropic không cung cấp Embeddings API → dùng OpenAI thay thế
        print("⚠️  Anthropic không có Embeddings API — đang dùng OpenAI embeddings thay thế.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )
