"""П4. Два настоящих клиента за неизменным контрактом generate()."""
import logging
import math
import re

import config
from contracts import LLMClient

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Ошибка доступности модели; не маскируется под содержательный ответ."""


def estimated_tokens(text: str) -> int:
    return math.ceil(len(text) / 3) if text else 0


class GigaChatClient:
    def __init__(self):
        from gigachat import GigaChat
        if not config.GIGACHAT_CREDENTIALS:
            raise LLMError("Не задан GIGACHAT_CREDENTIALS в локальном .env.")
        self.model = config.GIGACHAT_MODEL
        self._client = GigaChat(credentials=config.GIGACHAT_CREDENTIALS,
                                model=self.model,
                                verify_ssl_certs=config.GIGACHAT_VERIFY_SSL,
                                timeout=120)
        self.last_usage = None

    def generate(self, messages: list[dict], temperature: float = 0.3) -> str:
        self.last_usage = None
        try:
            response = self._client.chat({
                "model": self.model, "messages": messages,
                "temperature": max(0.01, temperature), "max_tokens": 600,
            })
            answer = response.choices[0].message.content
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("empty response")
            if response.usage is not None:
                self.last_usage = {"prompt_tokens": response.usage.prompt_tokens,
                                   "completion_tokens": response.usage.completion_tokens}
            return answer.strip()
        except Exception as exc:
            # Ни текст ошибки SDK, ни объект запроса не логируются: там могут быть ключи.
            log.warning("GigaChat: %s", type(exc).__name__)
            raise LLMError("GigaChat недоступен. Проверьте сеть, ключ и квоту; повторите запрос.") from None

    def count_tokens(self, text: str) -> int:
        try:
            return self._client.tokens_count([text], model=self.model)[0].tokens
        except Exception:
            return estimated_tokens(text)


class OllamaClient:
    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.host = config.OLLAMA_HOST.rstrip("/")
        self.last_usage = None

    def generate(self, messages: list[dict], temperature: float = 0.3) -> str:
        import requests
        self.last_usage = None
        try:
            response = requests.post(
                f"{self.host}/api/chat", json={
                    "model": self.model, "messages": messages, "stream": False, "think": False,
                    "options": {"temperature": max(0.01, temperature),
                                "seed": 37, "num_predict": 384, "num_ctx": 8192, "num_thread": 4},
                }, timeout=(10, 300))
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"]
            # Внутреннее рассуждение не является ответом и не нужно клиенту банка.
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
            if not content:
                raise ValueError("empty response")
            if all(isinstance(data.get(k), int) for k in ("prompt_eval_count", "eval_count")):
                self.last_usage = {"prompt_tokens": data["prompt_eval_count"],
                                   "completion_tokens": data["eval_count"]}
            return content
        except (requests.RequestException, ValueError, KeyError, TypeError):
            raise LLMError("Ollama недоступна или вернула неверный ответ. Запустите ollama serve и загрузите модель.") from None

    def count_tokens(self, text: str) -> int:
        # До запроса это только оценка. Реальный расход берётся из last_usage.
        return estimated_tokens(text)


def build_client() -> LLMClient:
    if config.LLM_BACKEND == "ollama":
        return OllamaClient()
    if config.LLM_BACKEND == "gigachat":
        return GigaChatClient()
    if config.LLM_BACKEND == "fake":
        from fake_llm import FakeClient
        return FakeClient()
    raise ValueError(f"Неизвестный LLM_BACKEND: {config.LLM_BACKEND!r}")
