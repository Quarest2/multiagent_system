"""
Клиент для OpenAI API.
"""

import openai
from typing import Optional
from .base_client import LLMClient


class OpenAIClient(LLMClient):
    """Клиент для OpenAI API."""

    def __init__(self, config):
        super().__init__(config)

        if not config.api_key:
            raise ValueError("API ключ OpenAI не указан")

        openai.api_key = config.api_key
        if config.base_url:
            openai.base_url = config.base_url

    def generate(self,
                 prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Генерация ответа через OpenAI API."""

        cache_key = f"{prompt}_{max_tokens}_{temperature}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            response = openai.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "Ты - эксперт по анализу данных и статистике."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                n=1
            )

            result = response.choices[0].message.content

            self.cache[cache_key] = result

            return result

        except Exception as e:
            raise Exception(f"Ошибка OpenAI API: {e}")