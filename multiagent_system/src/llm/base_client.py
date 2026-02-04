"""
Базовый клиент для работы с LLM.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import json
from ..config import LLMConfig


class LLMClient(ABC):
    """Абстрактный класс клиента LLM."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.cache = {}

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Генерация ответа на промпт."""
        pass

    def chat(self, messages: list, **kwargs) -> str:
        """Чат-интерфейс."""
        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, **kwargs)

    def _messages_to_prompt(self, messages: list) -> str:
        """Конвертация сообщений в промпт."""
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt += f"{role}: {content}\n\n"
        return prompt

    def clear_cache(self):
        """Очистка кэша."""
        self.cache = {}