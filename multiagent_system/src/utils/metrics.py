"""Сбор метрик системы."""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class SystemMetrics:
    """Метрики работы системы."""
    total_hypotheses: int = 0
    significant_hypotheses: int = 0
    avg_quality_score: float = 0.0
    avg_confidence: float = 0.0
    processing_time: float = 0.0
    refinement_iterations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            'total_hypotheses': self.total_hypotheses,
            'significant_hypotheses': self.significant_hypotheses,
            'significance_rate': (
                self.significant_hypotheses / self.total_hypotheses * 100
                if self.total_hypotheses > 0 else 0
            ),
            'avg_quality_score': round(self.avg_quality_score, 3),
            'avg_confidence': round(self.avg_confidence, 3),
            'processing_time': round(self.processing_time, 2),
            'refinement_iterations': self.refinement_iterations
        }
