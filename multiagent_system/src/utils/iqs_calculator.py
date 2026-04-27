"""Расчет Insight Quality Score (IQS)."""

import numpy as np
from typing import List, Dict, Any


class InsightQualityScore:
    """
    Оценка качества инсайтов (0-100 баллов).

    Компоненты:
    - Statistical Quality (30): статистическая строгость
    - Practical Value (25): практическая значимость
    - Diversity (20): разнообразие анализа
    - Novelty (15): новизна инсайтов
    - Efficiency (10): эффективность процесса
    """

    @staticmethod
    def calculate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Расчет IQS."""
        if not results:
            return {
                'total_score': 0,
                'grade': 'F',
                'breakdown': {},
                'recommendations': ['Нет результатов для анализа']
            }

        scores = {
            'statistical_quality': InsightQualityScore._statistical_quality(results),
            'practical_value': InsightQualityScore._practical_value(results),
            'diversity': InsightQualityScore._diversity(results),
            'novelty': InsightQualityScore._novelty(results),
            'efficiency': InsightQualityScore._efficiency(results)
        }

        total = sum(scores.values())
        grade = InsightQualityScore._get_grade(total)
        recommendations = InsightQualityScore._generate_recommendations(scores)

        return {
            'total_score': round(total, 1),
            'grade': grade,
            'breakdown': {k: round(v, 1) for k, v in scores.items()},
            'recommendations': recommendations
        }

    @staticmethod
    def _statistical_quality(results: List[Dict]) -> float:
        """Статистическое качество (30 баллов)."""
        score = 0
        significant = [r for r in results if r.get('is_significant', False)]

        # 1. Значимость (10 баллов) - не награждаем за 100%
        sig_rate = len(significant) / len(results)
        if 0.4 <= sig_rate <= 0.6:
            score += 10
        elif 0.3 <= sig_rate <= 0.7:
            score += 7
        else:
            score += min(10, sig_rate * 12)

        # 2. Сила эффекта (15 баллов)
        effect_sizes = []
        for r in significant:
            if r.get('hypothesis_type') == 'correlation':
                effect_sizes.append(abs(r.get('statistic', 0)))
            elif 'effect_size' in r.get('details', {}):
                effect_sizes.append(abs(r['details']['effect_size']))

        if effect_sizes:
            avg_effect = np.mean(effect_sizes)
            if avg_effect < 0.1:
                score += 3
            elif avg_effect < 0.3:
                score += 8
            elif avg_effect < 0.5:
                score += 12
            else:
                score += 15

        # 3. Размер выборки (5 баллов)
        avg_sample = np.mean([r.get('details', {}).get('sample_size', 0)
                              for r in results])
        if avg_sample > 1000:
            score += 5
        elif avg_sample > 100:
            score += 3
        else:
            score += 1

        return min(30, score)

    @staticmethod
    def _practical_value(results: List[Dict]) -> float:
        """Практическая ценность (25 баллов)."""
        score = 0

        # 1. Actionable variables (10 баллов)
        actionable_keywords = ['left', 'churn', 'salary', 'satisfaction',
                               'performance', 'price', 'revenue']
        actionable_count = sum(
            1 for r in results
            if r.get('is_significant') and
            any(kw in r.get('hypothesis_text', '').lower() for kw in actionable_keywords)
        )
        score += min(10, actionable_count * 2)

        # 2. Практическая значимость (10 баллов)
        practical_high = sum(
            1 for r in results
            if 'Высокая практическая значимость' in
            r.get('interpretation', {}).get('practical_significance', '')
        )
        score += min(10, practical_high * 3)

        # 3. Качество рекомендаций (5 баллов)
        avg_recs = np.mean([
            len(r.get('interpretation', {}).get('recommendations', []))
            for r in results if r.get('is_significant')
        ]) if any(r.get('is_significant') for r in results) else 0
        score += min(5, avg_recs)

        return min(25, score)

    @staticmethod
    def _diversity(results: List[Dict]) -> float:
        """Разнообразие (20 баллов)."""
        score = 0

        # 1. Типы гипотез (10 баллов)
        unique_types = len(set(r.get('hypothesis_type', '') for r in results))
        score += min(10, unique_types * 2)

        # 2. Методы анализа (5 баллов)
        unique_methods = len(set(r.get('method', '') for r in results))
        score += min(5, unique_methods)

        # 3. Покрытие переменных (5 баллов)
        # (упрощенная версия - считаем уникальные колонки в гипотезах)
        all_columns = set()
        for r in results:
            text = r.get('hypothesis_text', '')
            # Простая эвристика: извлекаем слова в кавычках
            import re
            cols = re.findall(r"'([^']+)'", text)
            all_columns.update(cols)

        coverage_score = min(5, len(all_columns) / 5)  # 5+ переменных = 5 баллов
        score += coverage_score

        return min(20, score)

    @staticmethod
    def _novelty(results: List[Dict]) -> float:
        """Новизна инсайтов (15 баллов)."""
        score = 0

        # 1. Избегание тривиальных корреляций (5 баллов)
        trivial_count = sum(
            1 for r in results
            if r.get('hypothesis_type') == 'correlation' and
            r.get('details', {}).get('correlation_strength') in ['очень слабая', 'слабая']
        )
        score += max(0, 5 - trivial_count * 0.5)

        # 2. Сложные паттерны (10 баллов)
        complex_types = ['segmentation', 'interaction_effect', 'mediation_analysis',
                         'threshold_effect', 'nonlinear_relationship']
        complex_count = sum(
            1 for r in results
            if r.get('hypothesis_type') in complex_types
        )
        score += min(10, complex_count * 2.5)

        return min(15, score)

    @staticmethod
    def _efficiency(results: List[Dict]) -> float:
        """Эффективность (10 баллов)."""
        score = 0

        # 1. Оптимальная доля значимых (5 баллов)
        sig_rate = sum(1 for r in results if r.get('is_significant')) / len(results)
        if 0.4 <= sig_rate <= 0.6:
            score += 5
        elif 0.3 <= sig_rate <= 0.7:
            score += 3
        else:
            score += 1

        # 2. Среднее качество (3 балла)
        avg_quality = np.mean([r.get('quality_score', 0) for r in results])
        score += avg_quality * 3

        # 3. Средняя уверенность (2 балла)
        avg_confidence = np.mean([r.get('confidence', 0) for r in results])
        score += avg_confidence * 2

        return min(10, score)

    @staticmethod
    def _get_grade(score: float) -> str:
        """Конвертация в оценку."""
        if score >= 85:
            return "A+ (Отлично)"
        elif score >= 75:
            return "A (Очень хорошо)"
        elif score >= 65:
            return "B (Хорошо)"
        elif score >= 55:
            return "C (Удовлетворительно)"
        elif score >= 45:
            return "D (Посредственно)"
        else:
            return "F (Неудовлетворительно)"

    @staticmethod
    def _generate_recommendations(scores: Dict[str, float]) -> List[str]:
        """Генерация рекомендаций."""
        recs = []

        if scores['statistical_quality'] < 20:
            recs.append("⚠ Улучшить статистическую строгость: проверить размер выборки и силу эффектов")

        if scores['practical_value'] < 15:
            recs.append("⚠ Фокусироваться на actionable переменных (churn, satisfaction, revenue)")

        if scores['diversity'] < 12:
            recs.append("⚠ Расширить типы анализа: добавить сегментацию, взаимодействия, нелинейные зависимости")

        if scores['novelty'] < 8:
            recs.append("⚠ Избегать тривиальных корреляций, искать сложные паттерны")

        if scores['efficiency'] < 6:
            recs.append("⚠ Оптимизировать долю значимых результатов (цель: 40-60%)")

        if not recs:
            recs.append("✓ Система работает эффективно")

        return recs
