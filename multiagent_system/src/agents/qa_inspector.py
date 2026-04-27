"""Агент проверки качества с AI."""

from typing import Dict, Any, List
import numpy as np
from ..config import AgentConfig, LLMConfig
from ..data.loader import Dataset
from ..llm.groq_client import GroqLLMClient
from ..utils.logger import logger


class QAInspector:
    """Инспектор качества с AI."""
    
    def __init__(self, config: AgentConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_client = GroqLLMClient(llm_config.api_key) if llm_config.enabled else None
    
    def inspect(self, hypothesis: Dict[str, Any], method: str,
                analysis_result: Dict[str, Any], dataset: Dataset) -> Dict[str, Any]:
        """Проверка качества анализа."""
        
        # Базовая проверка
        issues = []
        suggestions = []
        
        if 'error' in analysis_result:
            issues.append(f"Ошибка анализа: {analysis_result['error']}")
            return {
                'quality_score': 0.0,
                'issues': issues,
                'suggestions': ['Исправить ошибки'],
                'needs_refinement': True
            }
        
        # Проверка размера выборки
        sample_size = analysis_result.get('sample_size', 0)
        if sample_size < 30:
            issues.append(f"Маленькая выборка: {sample_size}")
            suggestions.append("Требуется больше данных")
        
        # Проверка граничных p-value
        p_value = analysis_result.get('p_value')
        if p_value and 0.04 < p_value < 0.06:
            issues.append("P-value близко к границе значимости")
            suggestions.append("Интерпретировать осторожно")
        
        # AI-проверка качества
        if self.config.use_ai_qa and self.llm_client and self.llm_client.is_available():
            ai_feedback = self._ai_quality_check(hypothesis, method, analysis_result)
            if ai_feedback:
                issues.extend(ai_feedback.get('issues', []))
                suggestions.extend(ai_feedback.get('suggestions', []))
                logger.debug("QA проверка усилена AI")
        
        quality_score = self._calculate_quality(analysis_result, issues)
        needs_refinement = quality_score < self.config.quality_threshold
        
        return {
            'quality_score': quality_score,
            'issues': issues,
            'suggestions': suggestions,
            'needs_refinement': needs_refinement
        }
    
    def _ai_quality_check(self, hypothesis: Dict, method: str, 
                         result: Dict) -> Optional[Dict]:
        """AI-проверка качества."""
        system_prompt = """Ты - строгий рецензент статистических исследований.
Находи методологические проблемы и предлагай улучшения."""
        
        user_prompt = f"""Оцени качество статистического анализа:

ГИПОТЕЗА: {hypothesis['text']}
МЕТОД: {method}
P-VALUE: {result.get('p_value', 'N/A')}
РАЗМЕР ВЫБОРКИ: {result.get('sample_size', 'N/A')}

Найди проблемы и предложи улучшения. Верни JSON:
{{
  "issues": ["проблема1", "проблема2"],
  "suggestions": ["предложение1", "предложение2"],
  "severity": "low|medium|high"
}}
"""
        
        return self.llm_client.generate_json(user_prompt, system_prompt)
    
    def _calculate_quality(self, result: Dict, issues: List[str]) -> float:
        """Расчет оценки качества."""
        if 'error' in result:
            return 0.0
        
        base_score = 1.0
        base_score -= len(issues) * 0.1
        
        if result.get('is_significant', False):
            base_score += 0.2
        
        sample_size = result.get('sample_size', 0)
        if sample_size > 100:
            base_score += 0.1
        
        return max(0.0, min(base_score, 1.0))
