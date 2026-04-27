"""Конфигурация системы."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Конфигурация LLM."""
    enabled: bool = True  # Включить AI агентов
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000


@dataclass
class AgentConfig:
    """Конфигурация агентов."""
    max_hypotheses: int = 100
    refinement_cycles: int = 5
    significance_level: float = 0.05
    quality_threshold: float = 0.7
    use_ai_generation: bool = True  # AI генерация гипотез
    use_ai_interpretation: bool = True  # AI интерпретация
    use_ai_qa: bool = True  # AI проверка качества


@dataclass
class DataConfig:
    """Конфигурация данных."""
    max_rows: int = 100000
    max_columns: int = 500
    missing_threshold: float = 0.3


@dataclass
class SystemConfig:
    """Общая конфигурация."""
    llm: LLMConfig
    agent: AgentConfig
    data: DataConfig
    
    @classmethod
    def default(cls):
        """Конфигурация по умолчанию."""
        return cls(
            llm=LLMConfig(),
            agent=AgentConfig(),
            data=DataConfig()
        )
