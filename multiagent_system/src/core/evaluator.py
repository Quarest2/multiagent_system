"""
Оценка качества гипотез и результатов.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import numpy as np

from ..config import ConfigManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class QualityDimensions(Enum):
    """Измерения качества (как в MCPybarra)."""
    FUNCTIONALITY = "functionality"  # Функциональность
    ROBUSTNESS = "robustness"  # Устойчивость
    SIGNIFICANCE = "significance"  # Статистическая значимость
    NOVELTY = "novelty"  # Новизна
    INTERPRETABILITY = "interpretability"  # Интерпретируемость


@dataclass
class QualityScore:
    """Оценка качества по измерениям."""
    functionality: float = 0.0
    robustness: float = 0.0
    significance: float = 0.0
    novelty: float = 0.0
    interpretability: float = 0.0

    @property
    def total(self) -> float:
        """Общая оценка с весами."""
        weights = {
            QualityDimensions.FUNCTIONALITY: 0.25,
            QualityDimensions.ROBUSTNESS: 0.20,
            QualityDimensions.SIGNIFICANCE: 0.25,
            QualityDimensions.NOVELTY: 0.15,
            QualityDimensions.INTERPRETABILITY: 0.15
        }

        return (
                self.functionality * weights[QualityDimensions.FUNCTIONALITY] +
                self.robustness * weights[QualityDimensions.ROBUSTNESS] +
                self.significance * weights[QualityDimensions.SIGNIFICANCE] +
                self.novelty * weights[QualityDimensions.NOVELTY] +
                self.interpretability * weights[QualityDimensions.INTERPRETABILITY]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'functionality': self.functionality,
            'robustness': self.robustness,
            'significance': self.significance,
            'novelty': self.novelty,
            'interpretability': self.interpretability,
            'total': self.total
        }


class QualityEvaluator:
    """Оценщик качества гипотез и результатов."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.agent_config = config.get_agent_config()

    def evaluate(self, analysis_result: Any) -> QualityScore:
        """
        Оценка качества результата анализа.

        Args:
            analysis_result: Результат анализа гипотезы

        Returns:
            Оценка качества
        """
        try:
            score = QualityScore()

            # 1. Функциональность (корректность метода)
            score.functionality = self._evaluate_functionality(analysis_result)

            # 2. Устойчивость (надежность)
            score.robustness = self._evaluate_robustness(analysis_result)

            # 3. Статистическая значимость
            score.significance = self._evaluate_significance(analysis_result)

            # 4. Новизна гипотезы
            score.novelty = self._evaluate_novelty(analysis_result)

            # 5. Интерпретируемость
            score.interpretability = self._evaluate_interpretability(analysis_result)

            return score

        except Exception as e:
            logger.error(f"Ошибка оценки качества: {e}")
            return QualityScore()  # Возвращаем нулевую оценку

    def _evaluate_functionality(self, result: Any) -> float:
        """Оценка функциональности (корректности метода)."""
        score = 0.0

        # Проверяем корректность метода
        if hasattr(result, 'method') and result.method != 'unknown':
            score += 0.3

        # Проверяем наличие статистики
        if hasattr(result, 'statistic') and result.statistic is not None:
            score += 0.2

        # Проверяем наличие p-value
        if hasattr(result, 'p_value') and result.p_value is not None:
            score += 0.2

        # Проверяем наличие доверительного интервала/эффекта
        if hasattr(result, 'effect_size') and result.effect_size is not None:
            score += 0.2

        # Проверяем отсутствие ошибок
        if hasattr(result, 'errors') and not result.errors:
            score += 0.1

        return min(score, 1.0)

    def _evaluate_robustness(self, result: Any) -> float:
        """Оценка устойчивости (надежности)."""
        score = 0.0

        # Учитываем количество циклов доработки (больше → надежнее)
        if hasattr(result, 'refinement_steps'):
            refinement_bonus = min(result.refinement_steps / 5, 0.2)
            score += refinement_bonus

        # Учитываем уверенность
        if hasattr(result, 'confidence'):
            score += result.confidence * 0.3

        # Учитываем качество данных
        if hasattr(result, 'data_quality'):
            score += getattr(result, 'data_quality', 0.5) * 0.2

        # Учитываем размер выборки (косвенно через наличие p-value)
        if hasattr(result, 'p_value') and result.p_value is not None:
            if result.p_value < self.agent_config.significance_level:
                score += 0.3  # Статистически значимый результат
            else:
                score += 0.1  # Незначимый, но корректно вычисленный

        return min(score, 1.0)

    def _evaluate_significance(self, result: Any) -> float:
        """Оценка статистической значимости."""
        if not hasattr(result, 'p_value') or result.p_value is None:
            return 0.0

        # Преобразуем p-value в оценку значимости
        p = result.p_value

        if p < 0.001:
            return 1.0  # Высокозначимый
        elif p < 0.01:
            return 0.8  # Очень значимый
        elif p < 0.05:
            return 0.6  # Значимый
        elif p < 0.1:
            return 0.3  # Гранично значимый
        else:
            return 0.1  # Незначимый

    def _evaluate_novelty(self, result: Any) -> float:
        """Оценка новизны гипотезы."""
        score = 0.0

        # Учитываем тип гипотезы
        if hasattr(result, 'hypothesis_type'):
            hypothesis_type = getattr(result, 'hypothesis_type', '').lower()

            # Более сложные/редкие типы получают более высокую оценку
            novelty_map = {
                'trend': 0.8,
                'interaction': 0.7,
                'nonlinear': 0.6,
                'cluster': 0.5,
                'association': 0.4,
                'mean_difference': 0.3,
                'correlation': 0.2,
                'distribution': 0.1
            }

            score += novelty_map.get(hypothesis_type, 0.1)

        # Учитываем количество задействованных переменных
        if hasattr(result, 'columns_involved'):
            columns = getattr(result, 'columns_involved', [])
            if len(columns) > 2:
                score += min((len(columns) - 2) * 0.1, 0.3)  # До +0.3 за 5+ переменных

        # Учитываем, была ли гипотеза улучшена LLM
        if hasattr(result, 'llm_enhanced') and result.llm_enhanced:
            score += 0.2

        return min(score, 1.0)

    def _evaluate_interpretability(self, result: Any) -> float:
        """Оценка интерпретируемости."""
        score = 0.0

        # Проверяем наличие заключения
        if hasattr(result, 'conclusion') and result.conclusion:
            conclusion = result.conclusion
            score += 0.3

            # Оцениваем длину и содержательность
            if len(conclusion) > 20:
                score += 0.1

        # Проверяем наличие объяснения
        if hasattr(result, 'explanation') and result.explanation:
            explanation = result.explanation
            score += 0.2

            # Оцениваем детальность объяснения
            if len(explanation) > 50:
                score += 0.1

        # Проверяем наличие рекомендаций
        if hasattr(result, 'recommendations') and result.recommendations:
            score += 0.1

        # Учитываем качество формулировки гипотезы
        if hasattr(result, 'hypothesis_text'):
            hypothesis = result.hypothesis_text
            if len(hypothesis.split()) > 5:  # Достаточно подробная
                score += 0.2

        return min(score, 1.0)

    def evaluate_batch(self, results: List[Any]) -> Dict[str, Any]:
        """
        Оценка качества пакета результатов.

        Args:
            results: Список результатов

        Returns:
            Статистика по качеству
        """
        if not results:
            return {}

        scores = [self.evaluate(r) for r in results]

        return {
            'count': len(results),
            'avg_total_quality': np.mean([s.total for s in scores]),
            'avg_functionality': np.mean([s.functionality for s in scores]),
            'avg_robustness': np.mean([s.robustness for s in scores]),
            'avg_significance': np.mean([s.significance for s in scores]),
            'avg_novelty': np.mean([s.novelty for s in scores]),
            'avg_interpretability': np.mean([s.interpretability for s in scores]),
            'high_quality_count': sum(1 for s in scores if s.total > 0.7),
            'low_quality_count': sum(1 for s in scores if s.total < 0.3),
            'quality_distribution': self._get_quality_distribution(scores)
        }

    def _get_quality_distribution(self, scores: List[QualityScore]) -> Dict[str, float]:
        """Распределение оценок качества."""
        if not scores:
            return {}

        total_scores = [s.total for s in scores]

        return {
            '0-0.2': sum(1 for s in total_scores if 0 <= s < 0.2) / len(total_scores),
            '0.2-0.4': sum(1 for s in total_scores if 0.2 <= s < 0.4) / len(total_scores),
            '0.4-0.6': sum(1 for s in total_scores if 0.4 <= s < 0.6) / len(total_scores),
            '0.6-0.8': sum(1 for s in total_scores if 0.6 <= s < 0.8) / len(total_scores),
            '0.8-1.0': sum(1 for s in total_scores if 0.8 <= s <= 1.0) / len(total_scores)
        }

    def get_improvement_recommendations(self, score: QualityScore) -> List[str]:
        """Рекомендации по улучшению качества."""
        recommendations = []

        if score.functionality < 0.5:
            recommendations.append(
                "Улучшить выбор статистического метода и расчет показателей"
            )

        if score.robustness < 0.5:
            recommendations.append(
                "Увеличить количество циклов доработки и проверок"
            )

        if score.significance < 0.5:
            recommendations.append(
                "Сфокусироваться на более значимых закономерностях"
            )

        if score.novelty < 0.5:
            recommendations.append(
                "Исследовать более сложные и неочевидные гипотезы"
            )

        if score.interpretability < 0.5:
            recommendations.append(
                "Улучшить объяснение результатов и выводов"
            )

        return recommendations