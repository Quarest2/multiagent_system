"""
Пакет агентов мультиагентной системы.
"""

from .hypothesis_generator import HypothesisGenerator
from .method_selector import MethodSelector
from .analysis_executor import AnalysisExecutor
from .interpreter import Interpreter
from .qa_inspector import QAInspector
from .refiner import Refiner

__all__ = [
    'HypothesisGenerator',
    'MethodSelector',
    'AnalysisExecutor',
    'Interpreter',
    'QAInspector',
    'Refiner'
]