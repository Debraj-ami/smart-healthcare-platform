import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def load_env_file(env_path: Optional[Union[str, Path]] = None, override: bool = False) -> None:
    """
    Loads environment variables from .env file into os.environ.
    Uses python-dotenv if installed, with a lightweight fallback parser.
    """
    if env_path is None:
        target = Path(__file__).resolve().parent.parent / ".env"
    else:
        target = Path(env_path)

    if not target.is_file():
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=target, override=override)
    except ImportError:
        try:
            with open(target, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and (override or key not in os.environ):
                            os.environ[key] = val
        except Exception as e:
            logger.warning("Failed to parse .env file at %s: %s", target, e)


# Automatically load .env configuration on module import
load_env_file()


class LLMError(Exception):
    """Base exception for all LLM client failures."""
    pass


class LLMUnavailableError(LLMError):
    """Raised when the LLM service cannot be reached, times out, or returns a 5xx gateway error."""
    pass


class LLMResponseError(LLMError):
    """Raised when the LLM returns an invalid, malformed, or unprocessable response."""
    pass


class BaseLLMClient(ABC):
    """Abstract interface for LLM provider implementations."""

    @abstractmethod
    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates completion from a list of dialogue messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            system_prompt: Optional overarching system instructions.
            temperature: Sampling temperature (default 0.2 for conservative extraction).
            response_format: Optional response format (e.g., {"type": "json_object"}).

        Returns:
            Raw assistant message text content.
            
        Raises:
            LLMUnavailableError: If provider is unreachable, times out, or returns 5xx.
            LLMResponseError: If provider returns a 4xx client error or unparseable payload.
        """
        pass

    def stream_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        """
        Default stream implementation: generates the full chat response and yields it.
        Subclasses (like OpenAICompatibleLLMClient) override this for true token streaming.
        """
        resp = self.generate_chat_response(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            response_format=response_format
        )
        yield resp

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Backward-compatible single-prompt completion method.
        """
        return self.generate_chat_response(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt
        )


class MockLLMClient(BaseLLMClient):
    """
    Deterministic mock LLM client for unit tests and local development without active LLM.
    Clearly labeled: development/testing extraction — not a medical NLP model.
    """

    def __init__(self, canned_response: Optional[str] = None):
        self.label = "development/testing extraction -- not a medical NLP model"
        self.canned_response = canned_response

    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        if self.canned_response is not None:
            return self.canned_response

        # Default deterministic mock completion
        return (
            "[Mock LLM Response: development/testing extraction — not a medical NLP model] "
            "I have processed your health message."
        )

    def stream_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        response = self.generate_chat_response(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            response_format=response_format
        )
        chunk_size = 15
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size]


class OpenAICompatibleLLMClient(BaseLLMClient):
    """
    Client for OpenAI-compatible LLM endpoints (OpenAI, Ollama, vLLM, Azure, LocalAI, etc.).
    Endpoint and credentials are dynamically loaded from environment variables or constructor args.
    Never hardcodes API keys or secrets.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model_name = (
            model_name
            or os.getenv("LLM_MODEL")
            or os.getenv("LLM_MODEL_NAME")
            or "gpt-4o-mini"
        )
        self.provider_label = (
            "ollama" if ("11434" in self.base_url or os.getenv("LLM_PROVIDER", "").lower() == "ollama")
            else os.getenv("LLM_PROVIDER", "openai_compatible").lower()
        )
        if timeout is not None:
            self.timeout = float(timeout)
        else:
            try:
                self.timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60.0"))
            except (ValueError, TypeError):
                self.timeout = 60.0

        if not self.api_key:
            raise ValueError(
                "OpenAICompatibleLLMClient requires an API key provided via argument or LLM_API_KEY environment variable."
            )

    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        import time
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": full_messages,
            "temperature": temperature
        }
        if response_format:
            payload["response_format"] = response_format

        start_time = time.perf_counter()
        provider = self.provider_label
        model = self.model_name

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )

            duration_sec = time.perf_counter() - start_time

            # Check HTTP status codes
            if response.status_code >= 500:
                logger.warning(
                    "LLM request failed: HTTP_5XX [provider=%s, model=%s, status=%d, duration=%.2fs]",
                    provider, model, response.status_code, duration_sec
                )
                raise LLMUnavailableError(
                    f"LLM upstream service unavailable (HTTP {response.status_code}) after {duration_sec:.2f}s."
                )
            if response.status_code >= 400:
                logger.warning(
                    "LLM request failed: HTTP_4XX [provider=%s, model=%s, status=%d, duration=%.2fs]",
                    provider, model, response.status_code, duration_sec
                )
                raise LLMResponseError(
                    f"LLM request rejected (HTTP {response.status_code}) after {duration_sec:.2f}s."
                )

            try:
                data = response.json()
            except Exception as e:
                logger.warning(
                    "LLM request failed: MALFORMED_RESPONSE (JSON decode failed) [provider=%s, model=%s, duration=%.2fs]: %s",
                    provider, model, duration_sec, e
                )
                raise LLMResponseError(
                    f"LLM returned invalid non-JSON payload after {duration_sec:.2f}s: {e}"
                ) from e

            choices = data.get("choices", [])
            if not choices:
                logger.warning(
                    "LLM request failed: MALFORMED_RESPONSE (no choices) [provider=%s, model=%s, duration=%.2fs]",
                    provider, model, duration_sec
                )
                raise LLMResponseError("LLM response returned no completion choices.")

            content = choices[0].get("message", {}).get("content")
            if content is None:
                logger.warning(
                    "LLM request failed: MALFORMED_RESPONSE (null content) [provider=%s, model=%s, duration=%.2fs]",
                    provider, model, duration_sec
                )
                raise LLMResponseError("LLM choice message contained no text content.")

            logger.info(
                "LLM request successful [provider=%s, model=%s, duration=%.2fs]",
                provider, model, duration_sec
            )
            return content.strip()

        except httpx.ConnectTimeout as e:
            duration_sec = time.perf_counter() - start_time
            logger.warning(
                "LLM request failed: CONNECT_TIMEOUT [provider=%s, model=%s, duration=%.2fs, timeout=%.1fs]",
                provider, model, duration_sec, self.timeout
            )
            raise LLMUnavailableError(
                f"LLM provider connection timed out after {duration_sec:.2f}s (configured connect timeout: {self.timeout}s)."
            ) from e

        except httpx.ReadTimeout as e:
            duration_sec = time.perf_counter() - start_time
            logger.warning(
                "LLM request failed: READ_TIMEOUT [provider=%s, model=%s, duration=%.2fs, timeout=%.1fs]",
                provider, model, duration_sec, self.timeout
            )
            raise LLMUnavailableError(
                f"LLM provider at {self.base_url} did not respond within {duration_sec:.2f}s (configured read timeout: {self.timeout}s). The model inference may be taking longer than the configured timeout limit."
            ) from e

        except httpx.TimeoutException as e:
            duration_sec = time.perf_counter() - start_time
            logger.warning(
                "LLM request failed: GENERIC_TIMEOUT [provider=%s, model=%s, duration=%.2fs, timeout=%.1fs]: %s",
                provider, model, duration_sec, self.timeout, e
            )
            raise LLMUnavailableError(
                f"LLM request timed out after {duration_sec:.2f}s: {e}"
            ) from e

        except (httpx.ConnectError, httpx.NetworkError) as e:
            duration_sec = time.perf_counter() - start_time
            logger.warning(
                "LLM request failed: CONNECTION_FAILURE [provider=%s, model=%s, duration=%.2fs]: %s",
                provider, model, duration_sec, e
            )
            raise LLMUnavailableError(
                f"Could not connect to LLM provider at {self.base_url} after {duration_sec:.2f}s: {e}"
            ) from e

        except (LLMUnavailableError, LLMResponseError):
            raise

        except Exception as e:
            duration_sec = time.perf_counter() - start_time
            logger.error(
                "LLM request failed: UNEXPECTED_ERROR [provider=%s, model=%s, duration=%.2fs]: %s",
                provider, model, duration_sec, e, exc_info=True
            )
            raise LLMResponseError(
                f"Unexpected error processing LLM response after {duration_sec:.2f}s: {e}"
            ) from e

    def stream_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        import time
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": full_messages,
            "temperature": temperature,
            "stream": True
        }
        if response_format:
            payload["response_format"] = response_format

        start_time = time.perf_counter()
        provider = self.provider_label
        model = self.model_name
        first_chunk_received = False
        prompt_chars_est = sum(len(m.get("content", "")) for m in full_messages)
        total_chars_emitted = 0

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status_code >= 500:
                        raise LLMUnavailableError(
                            f"LLM upstream service unavailable (HTTP {response.status_code})."
                        )
                    if response.status_code >= 400:
                        raise LLMResponseError(
                            f"LLM request rejected (HTTP {response.status_code})."
                        )

                    for line in response.iter_lines():
                        if not line:
                            continue
                        line_str = line.strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_str)
                                choices = chunk_json.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    delta_content = delta.get("content")
                                    if delta_content:
                                        if not first_chunk_received:
                                            ttft = time.perf_counter() - start_time
                                            logger.info(
                                                "LLM first token received [provider=%s, model=%s, ttft=%.2fs, prompt_chars=%d]",
                                                provider, model, ttft, prompt_chars_est
                                            )
                                            first_chunk_received = True
                                        total_chars_emitted += len(delta_content)
                                        yield delta_content
                            except json.JSONDecodeError:
                                continue

            duration_sec = time.perf_counter() - start_time
            logger.info(
                "LLM streaming request successful [provider=%s, model=%s, duration=%.2fs, response_chars=%d]",
                provider, model, duration_sec, total_chars_emitted
            )

        except httpx.ConnectTimeout as e:
            duration_sec = time.perf_counter() - start_time
            logger.warning(
                "LLM streaming request failed: CONNECT_TIMEOUT [provider=%s, model=%s, duration=%.2fs, timeout=%.1fs]",
                provider, model, duration_sec, self.timeout
            )
            raise LLMUnavailableError(
                f"LLM provider connection timed out after {duration_sec:.2f}s."
            ) from e

        except httpx.ReadTimeout as e:
            duration_sec = time.perf_counter() - start_time
            logger.warning(
                "LLM streaming request failed: READ_TIMEOUT [provider=%s, model=%s, duration=%.2fs, timeout=%.1fs]",
                provider, model, duration_sec, self.timeout
            )
            raise LLMUnavailableError(
                f"LLM provider at {self.base_url} did not respond within {duration_sec:.2f}s."
            ) from e

        except (httpx.ConnectError, httpx.NetworkError) as e:
            duration_sec = time.perf_counter() - start_time
            logger.warning(
                "LLM streaming request failed: CONNECTION_FAILURE [provider=%s, model=%s, duration=%.2fs]: %s",
                provider, model, duration_sec, e
            )
            raise LLMUnavailableError(
                f"Could not connect to LLM provider at {self.base_url} after {duration_sec:.2f}s: {e}"
            ) from e

        except (LLMUnavailableError, LLMResponseError):
            raise

        except Exception as e:
            duration_sec = time.perf_counter() - start_time
            logger.error(
                "LLM streaming request failed: UNEXPECTED_ERROR [provider=%s, model=%s, duration=%.2fs]: %s",
                provider, model, duration_sec, e, exc_info=True
            )
            raise LLMResponseError(
                f"Unexpected error processing LLM stream after {duration_sec:.2f}s: {e}"
            ) from e


def get_llm_client() -> BaseLLMClient:
    """
    Factory function creating the configured LLM client based on environment variables.

    Supported configurations:
    1. Local Ollama:
       LLM_PROVIDER=ollama
       LLM_BASE_URL=http://localhost:11434/v1
       LLM_API_KEY=ollama (optional, defaults to 'ollama')
       LLM_MODEL=llama3 (optional, defaults to 'llama3')

    2. OpenAI-Compatible Provider:
       LLM_PROVIDER=openai_compatible (or 'openai')
       LLM_BASE_URL=https://api.openai.com/v1 (or provider endpoint)
       LLM_API_KEY=<secret>
       LLM_MODEL=gpt-4o-mini

    3. Mock / Local Development Mode (Default):
       LLM_PROVIDER=mock (or unspecified)
    """
    load_env_file()
    provider = os.getenv("LLM_PROVIDER", "mock").lower().strip()

    if provider == "ollama":
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").strip()
        api_key = os.getenv("LLM_API_KEY", "ollama").strip() or "ollama"
        model_name = os.getenv("LLM_MODEL", os.getenv("LLM_MODEL_NAME", "llama3")).strip()
        return OpenAICompatibleLLMClient(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )

    if provider in ("openai", "openai_compatible", "openai-compatible"):
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "LLM_PROVIDER is set to '%s' but LLM_API_KEY is empty. Falling back to MockLLMClient.",
                provider
            )
            return MockLLMClient()
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
        model_name = os.getenv("LLM_MODEL", os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")).strip()
        return OpenAICompatibleLLMClient(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )

    # Default to safe mock development mode
    return MockLLMClient()
