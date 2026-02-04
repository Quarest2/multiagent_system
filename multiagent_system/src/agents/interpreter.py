"""
Интерпретатор результатов анализа.
"""

from typing import Dict, Any, Optional
import numpy as np

from ..config import AgentConfig
from ..llm.base_client import LLMClient
from ..llm.prompts import PromptManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class Interpreter:
    """Интерпретатор результатов статистического анализа."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_client = LLMClient(config) if config.enable_llm else None
        self.prompt_manager = PromptManager(config)

    def interpret(self,
                 hypothesis: Dict[str, Any],
                 analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Интерпретация результатов анализа.

        Args:
            hypothesis: Гипотеза
            analysis_result: Результаты анализа

        Returns:
            Интерпретация
        """
        if analysis_result.get('error'):
            return self._error_interpretation(hypothesis, analysis_result)

        base_interpretation = self._base_interpretation(hypothesis, analysis_result)

        if self.config.enable_llm and self.llm_client:
            enhanced = self._llm_enhancement(hypothesis, analysis_result, base_interpretation)
            if enhanced:
                return enhanced

        return base_interpretation

    def _base_interpretation(self,
                            hypothesis: Dict[str, Any],
                            analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Базовая интерпретация без LLM."""
        p_value = analysis_result.get('p_value')
        is_significant = analysis_result.get('is_significant', False)

        if p_value is None:
            conclusion = "Не удалось провести анализ"
            confidence = 0.0
        elif p_value < self.config.significance_level:
            conclusion = f"Гипотеза подтверждена (p={p_value:.4f} < {self.config.significance_level})"
            confidence = 1.0 - p_value
        else:
            conclusion = f"Гипотеза отклонена (p={p_value:.4f} ≥ {self.config.significance_level})"
            confidence = p_value

        explanation = self._generate_explanation(hypothesis, analysis_result, conclusion)

        confidence = self._calculate_confidence(analysis_result)

        return {
            'conclusion': conclusion,
            'explanation': explanation,
            'confidence': confidence,
            'is_llm_enhanced': False
        }

    def _generate_explanation(self,
                             hypothesis: Dict[str, Any],
                             analysis_result: Dict[str, Any],
                             conclusion: str) -> str:
        """Генерация объяснения результатов."""
        method = analysis_result.get('method', 'unknown')
        p_value = analysis_result.get('p_value')
        effect_size = analysis_result.get('effect_size')

        explanation_parts = []

        explanation_parts.append(conclusion)

        if method != 'unknown':
            method_descriptions = {
                'pearson': "коэффициент корреляции Пирсона",
                'spearman': "ранговая корреляция Спирмена",
                't_test_ind': "t-тест для независимых выборок",
                'chi2': "критерий хи-квадрат",
                'shapiro': "тест Шапиро-Уилка"
            }
            method_desc = method_descriptions.get(method, method)
            explanation_parts.append(f"Использован {method_desc}.")

        if effect_size is not None:
            if abs(effect_size) < 0.1:
                effect_desc = "очень маленький"
            elif abs(effect_size) < 0.3:
                effect_desc = "маленький"
            elif abs(effect_size) < 0.5:
                effect_desc = "средний"
            else:
                effect_desc = "большой"

            if effect_desc:
                explanation_parts.append(f"Размер эффекта: {effect_desc} ({effect_size:.3f}).")

        if p_value is not None:
            if p_value < 0.001:
                explanation_parts.append("Результат имеет высокую статистическую значимость.")
            elif p_value < 0.01:
                explanation_parts.append("Результат статистически значим.")

        return " ".join(explanation_parts)

    def _calculate_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Расчет уверенности в результатах."""
        confidence = 0.5

        p_value = analysis_result.get('p_value')
        if p_value is not None:
            if p_value < 0.001:
                confidence += 0.3
            elif p_value < 0.01:
                confidence += 0.2
            elif p_value < 0.05:
                confidence += 0.1
            elif p_value > 0.2:
                confidence -= 0.1

        effect_size = analysis_result.get('effect_size')
        if effect_size is not None:
            if abs(effect_size) > 0.5:
                confidence += 0.2
            elif abs(effect_size) > 0.3:
                confidence += 0.1

        n = analysis_result.get('n')
        if n is not None:
            if n > 1000:
                confidence += 0.1
            elif n > 100:
                confidence += 0.05
            elif n < 30:
                confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _llm_enhancement(self,
                        hypothesis: Dict[str, Any],
                        analysis_result: Dict[str, Any],
                        base_interpretation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Улучшение интерпретации через LLM."""
        try:
            prompt = self.prompt_manager.get_prompt(
                'interpretation',
                hypothesis_text=hypothesis['text'],
                method=analysis_result.get('method', 'unknown'),
                p_value=analysis_result.get('p_value'),
                statistic=analysis_result.get('statistic'),
                effect_size=analysis_result.get('effect_size'),
                confidence_interval=analysis_result.get('confidence_interval')
            )

            response = self.llm_client.generate(prompt, max_tokens=500)

            enhanced_interpretation = self._parse_llm_response(response, base_interpretation)
            enhanced_interpretation['is_llm_enhanced'] = True

            return enhanced_interpretation

        except Exception as e:
            logger.error(f"Ошибка улучшения интерпретации через LLM: {e}")
            return None

    def _parse_llm_response(self,
                           llm_response: str,
                           base_interpretation: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг ответа LLM."""
        lines = llm_response.strip().split('\n')

        conclusion = base_interpretation['conclusion']
        explanation = base_interpretation['explanation']

        for line in lines:
            line_lower = line.lower()

            if 'вывод' in line_lower and len(line) > 10:
                conclusion = line.strip()
            elif 'объяснение' in line_lower or 'значение' in line_lower:
                if len(line) > 20:
                    explanation = line.strip()

        return {
            'conclusion': conclusion,
            'explanation': explanation,
            'confidence': base_interpretation['confidence'],
            'is_llm_enhanced': True
        }

    def _error_interpretation(self,
                             hypothesis: Dict[str, Any],
                             analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Интерпретация при ошибке анализа."""
        error_msg = analysis_result.get('error', 'Неизвестная ошибка')

        return {
            'conclusion': f"Ошибка анализа: {error_msg}",
            'explanation': "Не удалось проверить гипотезу из-за ошибки в анализе.",
            'confidence': 0.0,
            'is_llm_enhanced': False
        }