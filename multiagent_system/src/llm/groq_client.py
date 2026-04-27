"""Клиент для работы с Groq API (бесплатный LLM)."""

import os
import json
from typing import Optional, Dict, Any
from groq import Groq
from ..utils.logger import logger


class GroqLLMClient:
    """Клиент для Groq API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация клиента.
        
        Args:
            api_key: API ключ Groq (или через GROQ_API_KEY в .env)
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY не найден. AI функции отключены.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            logger.info("Groq LLM клиент инициализирован")
    
    def generate(self, prompt: str, system_prompt: str = None, 
                 max_tokens: int = 2000, temperature: float = 0.7) -> Optional[str]:
        """
        Генерация ответа от LLM.
        
        Args:
            prompt: Пользовательский промпт
            system_prompt: Системный промпт
            max_tokens: Максимум токенов
            temperature: Температура генерации
            
        Returns:
            Ответ LLM или None при ошибке
        """
        if not self.client:
            return None
        
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Бесплатная модель
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Ошибка Groq API: {e}")
            return None
    
    def generate_json(self, prompt: str, system_prompt: str = None) -> Optional[Dict]:
        """Генерация JSON ответа."""
        response = self.generate(prompt, system_prompt, temperature=0.3)
        
        if not response:
            return None
        
        try:
            # Извлекаем JSON из ответа
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        
        except Exception as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            return None
    
    def is_available(self) -> bool:
        """Проверка доступности API."""
        return self.client is not None
