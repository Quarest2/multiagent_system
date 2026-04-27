"""Оценка качества результатов."""

from typing import Dict, Any
import numpy as np
from ..config import AgentConfig


class QualityEvaluator:
    """Оценщик качества результатов анализа."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def evaluate(self, analysis_result: Dict[str, Any]) -> float:
        """Комплексная оценка качества."""

        if 'error' in analysis_result:
            return 0.0

        scores = []

        # 1. Функциональность (есть ли результат)
        if 'p_value' in analysis_result and 'statistic' in analysis_result:
            scores.append(0.3)

        # 2. Статистическая значимость
        if analysis_result.get('is_significant', False):
            scores.append(0.3)
        else:
            scores.append(0.1)

        # 3. Размер выборки
        sample_size = analysis_result.get('sample_size', 0)
        if sample_size > 100:
            scores.append(0.2)
        elif sample_size > 30:
            scores.append(0.1)
        else:
            scores.append(0.05)

        # 4. Качество p-value (не слишком близко к границе)
        p_value = analysis_result.get('p_value', 1.0)
        if p_value < 0.01 or p_value > 0.1:
            scores.append(0.2)
        else:
            scores.append(0.1)

        return min(sum(scores), 1.0)
