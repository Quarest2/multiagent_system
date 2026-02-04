"""
Конфигурация системы.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
from pathlib import Path
import yaml


@dataclass
class LLMConfig:
    """Конфигурация LLM."""
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class AgentConfig:
    """Конфигурация агентов."""
    max_hypotheses: int = 15
    refinement_cycles: int = 3
    significance_level: float = 0.05
    quality_threshold: float = 0.7
    enable_llm: bool = True


@dataclass
class DataConfig:
    """Конфигурация данных."""
    max_rows: int = 10000
    max_columns: int = 50
    missing_threshold: float = 0.3
    sample_size: int = 1000


@dataclass
class SystemConfig:
    """Основная конфигурация системы."""
    llm: LLMConfig
    agents: AgentConfig
    data: DataConfig
    cache_dir: str = "./cache"
    log_level: str = "INFO"
    output_dir: str = "./outputs"

    @classmethod
    def from_yaml(cls, config_path: str = "config/default.yaml") -> "SystemConfig":
        """Загрузка конфигурации из YAML."""
        config_path = Path(config_path)

        if not config_path.exists():
            return cls(
                llm=LLMConfig(),
                agents=AgentConfig(),
                data=DataConfig()
            )

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        llm_config = LLMConfig(
            **config_dict.get('llm', {}),
            api_key=os.getenv('OPENAI_API_KEY') or config_dict.get('llm', {}).get('api_key')
        )

        return cls(
            llm=llm_config,
            agents=AgentConfig(**config_dict.get('agents', {})),
            data=DataConfig(**config_dict.get('data', {})),
            **config_dict.get('system', {})
        )


class ConfigManager:
    """Менеджер конфигурации."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = SystemConfig.from_yaml(config_path) if config_path else SystemConfig()
        self._setup_directories()

    def _setup_directories(self):
        """Создание необходимых директорий."""
        dirs = [
            self.config.cache_dir,
            self.config.output_dir,
            f"{self.config.output_dir}/reports",
            f"{self.config.output_dir}/visualizations",
            f"{self.config.cache_dir}/llm_responses"
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get_llm_config(self) -> LLMConfig:
        return self.config.llm

    def get_agent_config(self) -> AgentConfig:
        return self.config.agents

    def get_data_config(self) -> DataConfig:
        return self.config.data

    def update_llm_config(self, **kwargs):
        """Обновление конфигурации LLM."""
        for key, value in kwargs.items():
            if hasattr(self.config.llm, key):
                setattr(self.config.llm, key, value)

    def save_config(self, path: str):
        """Сохранение конфигурации."""
        config_dict = {
            'llm': self.config.llm.__dict__,
            'agents': self.config.agents.__dict__,
            'data': self.config.data.__dict__,
            'system': {
                'cache_dir': self.config.cache_dir,
                'log_level': self.config.log_level,
                'output_dir': self.config.output_dir
            }
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)