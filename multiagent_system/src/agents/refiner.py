"""Агент доработки гипотез."""

from typing import Dict, Any
from ..config import AgentConfig
from ..utils.logger import logger


class Refiner:
    """Доработчик гипотез на основе обратной связи."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def refine(self, hypothesis: Dict[str, Any],
               qa_report: Dict[str, Any]) -> Dict[str, Any]:
        """Доработка гипотезы."""

        if not qa_report.get('needs_refinement', False):
            # Доработка не требуется
            return hypothesis

        refined = hypothesis.copy()
        issues = qa_report.get('issues', [])

        # Анализируем проблемы и корректируем
        for issue in issues:
            if 'Маленькая выборка' in issue:
                # Помечаем гипотезу как требующую большей выборки
                refined['note'] = 'Требуется больше данных'

            elif 'P-value близко' in issue:
                # Снижаем уверенность
                refined['confidence_penalty'] = 0.1

        # Увеличиваем счетчик доработок
        refined['refinement_count'] = refined.get('refinement_count', 0) + 1

        logger.debug(f"Гипотеза {hypothesis['id']} доработана")

        return refined
