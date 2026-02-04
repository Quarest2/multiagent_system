"""
Агент доработки гипотез на основе обратной связи (аналог Code Refiner из MCPybarra).
"""

from typing import Dict, Any, List
import re

from ..config import AgentConfig
from ..llm.base_client import LLMClient
from ..llm.prompts import PromptManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class Refiner:
    """Агент доработки гипотез."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_client = LLMClient(config) if config.enable_llm else None
        self.prompt_manager = PromptManager(config)

    def refine(self, hypothesis: Dict[str, Any], feedback: List[str]) -> Dict[str, Any]:
        """
        Доработка гипотезы на основе обратной связи.

        Args:
            hypothesis: Исходная гипотеза
            feedback: Список замечаний и рекомендаций

        Returns:
            Улучшенная гипотеза
        """
        if not feedback or not self.llm_client:
            return hypothesis

        try:
            prompt = self.prompt_manager.get_prompt(
                'hypothesis_refinement',
                original_hypothesis=hypothesis['text'],
                hypothesis_type=hypothesis['type'],
                columns_involved=hypothesis.get('columns', []),
                feedback='\n'.join(feedback),
                refinement_goal="Улучшить формулировку, исправить методологические проблемы, уточнить условия"
            )

            response = self.llm_client.generate(prompt, max_tokens=500)

            refined_hypothesis = self._extract_refined_hypothesis(response, hypothesis)

            logger.info(f"Гипотеза {hypothesis['id']} доработана")
            return refined_hypothesis

        except Exception as e:
            logger.error(f"Ошибка доработки гипотезы: {e}")
            return hypothesis

    def _extract_refined_hypothesis(self,
                                    llm_response: str,
                                    original_hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение доработанной гипотезы из ответа LLM."""
        lines = llm_response.split('\n')

        refined_text = original_hypothesis['text']
        refined_type = original_hypothesis['type']

        keywords = ['гипотеза', 'предположение', 'утверждение', 'hypothesis', 'assumption']
        improved_keywords = ['улучшенная', 'доработанная', 'скорректированная', 'refined', 'improved']

        for line in lines:
            line_lower = line.lower()

            has_hypothesis_keywords = any(kw in line_lower for kw in keywords)
            has_improved_keywords = any(kw in line_lower for kw in improved_keywords)

            if has_hypothesis_keywords or (has_improved_keywords and len(line.strip()) > 20):
                cleaned_line = re.sub(r'^[\d\-•*]\s*', '', line.strip())
                if len(cleaned_line) > len(refined_text) * 0.5:
                    refined_text = cleaned_line
                    break

        if 'корреляция' in refined_text.lower() or 'correlation' in refined_text.lower():
            refined_type = 'correlation'
        elif 'различие' in refined_text.lower() or 'difference' in refined_text.lower():
            refined_type = 'mean_difference'
        elif 'нормальн' in refined_text.lower() or 'normal' in refined_text.lower():
            refined_type = 'normality'
        elif 'независим' in refined_text.lower() or 'independent' in refined_text.lower():
            refined_type = 'independence'

        refined_hypothesis = original_hypothesis.copy()
        refined_hypothesis['text'] = refined_text
        refined_hypothesis['type'] = refined_type
        refined_hypothesis['refined'] = True
        refined_hypothesis['refinement_count'] = refined_hypothesis.get('refinement_count', 0) + 1

        return refined_hypothesis

    def refine_method(self,
                      hypothesis: Dict[str, Any],
                      current_method: str,
                      feedback: List[str]) -> str:
        """
        Доработка метода анализа на основе обратной связи.

        Args:
            hypothesis: Гипотеза
            current_method: Текущий метод
            feedback: Обратная связь

        Returns:
            Улучшенный метод
        """
        if not feedback or 'неподходящий метод' not in ' '.join(feedback).lower():
            return current_method

        method_alternatives = {
            't_test_ind': ['mannwhitneyu', 'anova'],
            'pearson': ['spearman', 'kendall'],
            'chi2': ['fisher_exact'],
            'shapiro': ['normaltest', 'anderson']
        }

        if current_method in method_alternatives:
            return method_alternatives[current_method][0]

        return current_method