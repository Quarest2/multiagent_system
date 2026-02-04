"""
Инспектор качества гипотез (аналог QA Inspector из MCPybarra).
"""

from typing import Dict, Any, List, Optional
import numpy as np

from ..config import AgentConfig
from ..data.loader import Dataset
from ..llm.base_client import LLMClient
from ..llm.prompts import PromptManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class QAInspector:
    """Инспектор качества гипотез."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_client = LLMClient(config) if config.enable_llm else None
        self.prompt_manager = PromptManager(config)

    def inspect(self,
                hypothesis: Dict[str, Any],
                method: str,
                analysis_result: Dict[str, Any],
                dataset: Dataset) -> Dict[str, Any]:
        """
        Проверка качества гипотезы и результатов анализа.

        Args:
            hypothesis: Гипотеза
            method: Использованный метод
            analysis_result: Результаты анализа
            dataset: Набор данных

        Returns:
            Отчет о проверке качества
        """
        report = {
            'hypothesis_id': hypothesis['id'],
            'method': method,
            'needs_refinement': False,
            'feedback': [],
            'scores': {},
            'recommendations': []
        }

        statistical_issues = self._check_statistical_assumptions(hypothesis, method, dataset)
        if statistical_issues:
            report['feedback'].extend(statistical_issues)
            report['needs_refinement'] = True

        data_issues = self._check_data_quality(hypothesis, dataset)
        if data_issues:
            report['feedback'].extend(data_issues)

        if analysis_result.get('p_value') is not None:
            interpretation_issues = self._check_interpretation(hypothesis, analysis_result)
            if interpretation_issues:
                report['feedback'].extend(interpretation_issues)

        if self.config.enable_llm and self.llm_client:
            llm_feedback = self._get_llm_feedback(hypothesis, method, analysis_result, dataset)
            report['feedback'].extend(llm_feedback.get('issues', []))
            report['recommendations'].extend(llm_feedback.get('recommendations', []))

            if llm_feedback.get('needs_refinement', False):
                report['needs_refinement'] = True

        report['scores'] = self._calculate_quality_scores(report, analysis_result)

        return report

    def _check_statistical_assumptions(self,
                                       hypothesis: Dict[str, Any],
                                       method: str,
                                       dataset: Dataset) -> List[str]:
        """Проверка статистических предположений метода."""
        issues = []
        df = dataset.df
        columns = hypothesis.get('columns', [])

        if method in ['t_test_ind', 'anova']:
            for col in columns:
                if col in dataset.metadata['numeric_columns']:
                    data = df[col].dropna()
                    if len(data) > 30:
                        skewness = abs(data.skew())
                        if skewness > 1:
                            issues.append(f"Распределение '{col}' имеет высокую асимметрию (skew={skewness:.2f})")

        elif method in ['pearson']:
            if len(columns) >= 2:
                col1, col2 = columns[0], columns[1]
                data1 = df[col1].dropna()
                data2 = df[col2].dropna()

                q1_1, q3_1 = np.percentile(data1, [25, 75])
                iqr_1 = q3_1 - q1_1
                outliers_1 = ((data1 < (q1_1 - 1.5 * iqr_1)) | (data1 > (q3_1 + 1.5 * iqr_1))).sum()

                if outliers_1 > len(data1) * 0.05:
                    issues.append(f"Переменная '{col1}' имеет много выбросов ({outliers_1})")

        return issues

    def _check_data_quality(self, hypothesis: Dict[str, Any], dataset: Dataset) -> List[str]:
        """Проверка качества данных для гипотезы."""
        issues = []
        columns = hypothesis.get('columns', [])

        for col in columns:
            if col not in dataset.df.columns:
                issues.append(f"Колонка '{col}' не найдена в данных")
                continue

            missing = dataset.df[col].isnull().sum()
            total = len(dataset.df)
            missing_pct = missing / total if total > 0 else 0

            if missing_pct > self.config.quality_threshold:
                issues.append(f"Колонка '{col}' имеет много пропусков ({missing_pct:.1%})")

            if col in dataset.metadata['numeric_columns']:
                data = dataset.df[col].dropna()
                if len(data) > 1 and data.std() == 0:
                    issues.append(f"Колонка '{col}' не имеет дисперсии (все значения одинаковы)")

        return issues

    def _check_interpretation(self,
                              hypothesis: Dict[str, Any],
                              analysis_result: Dict[str, Any]) -> List[str]:
        """Проверка корректности интерпретации результатов."""
        issues = []

        p_value = analysis_result.get('p_value')
        if p_value is not None:
            if 0.04 < p_value < 0.05:
                issues.append("P-value находится близко к границе значимости. Будьте осторожны с интерпретацией.")

            effect_size = analysis_result.get('effect_size')
            if effect_size is not None and abs(effect_size) < 0.1 and p_value < 0.05:
                issues.append(
                    "Статистически значимый, но очень маленький размер эффекта. Практическая значимость может быть низкой.")

        return issues

    def _get_llm_feedback(self,
                          hypothesis: Dict[str, Any],
                          method: str,
                          analysis_result: Dict[str, Any],
                          dataset: Dataset) -> Dict[str, Any]:
        """Получение обратной связи от LLM."""
        try:
            prompt = self.prompt_manager.get_prompt(
                'qa_inspection',
                hypothesis_text=hypothesis['text'],
                hypothesis_type=hypothesis['type'],
                method=method,
                p_value=analysis_result.get('p_value'),
                statistic=analysis_result.get('statistic'),
                dataset_summary=dataset.get_summary(),
                columns_involved=hypothesis.get('columns', [])
            )

            response = self.llm_client.generate(prompt, max_tokens=800)

            return self._parse_llm_qa_response(response)

        except Exception as e:
            logger.error(f"Ошибка получения обратной связи от LLM: {e}")
            return {'issues': [], 'recommendations': [], 'needs_refinement': False}

    def _parse_llm_qa_response(self, response: str) -> Dict[str, Any]:
        """Парсинг ответа LLM по проверке качества."""
        issues = []
        recommendations = []
        needs_refinement = False

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line_lower = line.lower()

            if 'проблема' in line_lower or 'issue' in line_lower:
                current_section = 'issues'
            elif 'рекомендация' in line_lower or 'recommendation' in line_lower:
                current_section = 'recommendations'
            elif 'доработка' in line_lower or 'refinement' in line_lower:
                if 'нужна' in line_lower or 'требуется' in line_lower:
                    needs_refinement = True

            if line.strip() and current_section:
                if current_section == 'issues' and len(line.strip()) > 10:
                    issues.append(line.strip())
                elif current_section == 'recommendations' and len(line.strip()) > 10:
                    recommendations.append(line.strip())

        return {
            'issues': issues,
            'recommendations': recommendations,
            'needs_refinement': needs_refinement
        }

    def _calculate_quality_scores(self,
                                  report: Dict[str, Any],
                                  analysis_result: Dict[str, Any]) -> Dict[str, float]:
        """Расчет оценок качества."""
        scores = {
            'data_quality': 1.0,
            'method_appropriateness': 1.0,
            'interpretation_quality': 1.0,
            'overall': 1.0
        }

        issues_count = len(report.get('feedback', []))
        penalty = min(issues_count * 0.1, 0.5)

        scores['data_quality'] -= penalty

        p_value = analysis_result.get('p_value')
        if p_value is not None:
            if p_value < 0.001:
                scores['interpretation_quality'] += 0.2
            elif p_value > 0.1:
                scores['interpretation_quality'] -= 0.1

        scores['overall'] = np.mean(list(scores.values()))

        return {k: max(0, min(1, v)) for k, v in scores.items()}