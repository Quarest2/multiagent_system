"""
Мультиагентная система генерации и проверки гипотез.
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"
__description__ = "Мультиагентная система для автоматической генерации и проверки статистических гипотез на основе табличных данных"

# Основные компоненты
from .core import Orchestrator, Workflow, QualityEvaluator
from .data import DataLoader, DataPreprocessor, DataValidator
from .agents import (
    HypothesisGenerator, MethodSelector, AnalysisExecutor,
    Interpreter, QAInspector, Refiner
)
from .llm import OpenAIClient, PromptManager, LLMCache
from .utils import (
    setup_logger, MetricsCollector,
    ResultVisualizer, ReportGenerator
)
from .config import ConfigManager

__all__ = [
    # Основные классы
    'Orchestrator',
    'Workflow',
    'QualityEvaluator',

    # Работа с данными
    'DataLoader',
    'DataPreprocessor',
    'DataValidator',

    # Агенты
    'HypothesisGenerator',
    'MethodSelector',
    'AnalysisExecutor',
    'Interpreter',
    'QAInspector',
    'Refiner',

    # LLM интеграция
    'OpenAIClient',
    'PromptManager',
    'LLMCache',

    # Утилиты
    'setup_logger',
    'MetricsCollector',
    'ResultVisualizer',
    'ReportGenerator',

    # Конфигурация
    'ConfigManager',

    # Метаданные
    '__version__',
    '__author__',
    '__email__',
    '__description__'
]