"""Агент интерпретации результатов с AI."""

from typing import Dict, Any, List, Optional
from ..config import AgentConfig, LLMConfig
from ..llm.groq_client import GroqLLMClient
from ..utils.logger import logger


class Interpreter:
    """Интерпретатор результатов анализа с AI."""

    def __init__(self, config: AgentConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_client = GroqLLMClient(llm_config.api_key) if llm_config.enabled else None
        self.ai_call_count = 0  # Счетчик для rate limiting
        self.max_ai_calls_per_run = 15  # Ограничение на бесплатном плане

    def interpret(self, hypothesis: Dict[str, Any],
                  analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Интерпретация результатов анализа."""

        # Базовая интерпретация
        base_interpretation = self._base_interpretation(hypothesis, analysis_result)

        # AI-усиление (если доступно и не превышен лимит)
        if (self.config.use_ai_interpretation and
                self.llm_client and
                self.llm_client.is_available() and
                self.ai_call_count < self.max_ai_calls_per_run):

            ai_interpretation = self._ai_interpretation(hypothesis, analysis_result)
            if ai_interpretation:
                base_interpretation['ai_explanation'] = ai_interpretation
                base_interpretation['explanation'] = ai_interpretation.get('explanation',
                                                                           base_interpretation['explanation'])
                logger.debug(f"Интерпретация усилена AI ({self.ai_call_count}/{self.max_ai_calls_per_run})")

        return base_interpretation

    def _base_interpretation(self, hypothesis: Dict, analysis_result: Dict) -> Dict[str, Any]:
        """Базовая интерпретация без AI."""
        p_value = analysis_result.get('p_value', 1.0)
        is_significant = analysis_result.get('is_significant', False)

        if is_significant:
            conclusion = f"✓ Гипотеза подтверждена (p={p_value:.4f})"
        else:
            conclusion = f"✗ Гипотеза отклонена (p={p_value:.4f})"

        # Расширенное объяснение в зависимости от метода
        method = analysis_result.get('method', 'unknown')
        explanation = self._generate_method_explanation(hypothesis, analysis_result, method)

        return {
            'conclusion': conclusion,
            'explanation': explanation,
            'practical_significance': self._assess_practical_significance(analysis_result),
            'recommendations': self._generate_recommendations(hypothesis, analysis_result)
        }

    def _generate_method_explanation(self, hypothesis: Dict, result: Dict, method: str) -> str:
        """Генерация объяснения в зависимости от метода."""
        p_value = result.get('p_value', 1.0)
        is_sig = result.get('is_significant', False)

        if method == 'segmentation':
            n_clusters = result.get('n_clusters', 0)
            silhouette = result.get('silhouette_score', 0)
            profiles = result.get('segment_profiles', [])

            if is_sig:
                expl = f"Обнаружено {n_clusters} различных сегмента (качество кластеризации: {silhouette:.2f}). "
                if profiles:
                    sizes = [p['size'] for p in profiles]
                    expl += f"Размеры сегментов: {', '.join(map(str, sizes))} записей. "
                    expl += "Сегменты значительно отличаются по характеристикам."
            else:
                expl = "Четкие сегменты не выявлены. Данные относительно однородны."
            return expl

        elif method == 'interaction':
            improvement = result.get('improvement', 0)
            r2_base = result.get('r2_base', 0)
            r2_int = result.get('r2_with_interaction', 0)

            if is_sig:
                expl = (f"Взаимодействие переменных значимо улучшает модель (R²: {r2_base:.3f} → {r2_int:.3f}, "
                        f"прирост {improvement:.3f}). Эффект одной переменной зависит от значения другой.")
            else:
                expl = "Взаимодействие переменных не дает значимого улучшения. Переменные действуют независимо."
            return expl

        elif method == 'threshold':
            thresholds = result.get('suggested_thresholds', [])

            if is_sig:
                expl = f"Обнаружены потенциальные пороговые значения: {', '.join(f'{t:.2f}' for t in thresholds)}. "
                expl += "Эффект может изменяться при переходе через эти границы."
            else:
                expl = "Четкие пороговые эффекты не обнаружены. Зависимость линейная или отсутствует."
            return expl

        elif method == 'nonlinear':
            r2_linear = result.get('r2_linear', 0)
            r2_poly = result.get('r2_polynomial', 0)
            improvement = result.get('improvement', 0)

            if is_sig:
                expl = (
                    f"Обнаружена нелинейная зависимость (полиномиальная модель лучше: R² {r2_linear:.3f} → {r2_poly:.3f}). "
                    f"Зависимость имеет U-образную или перевернутую U-форму.")
            else:
                expl = "Зависимость близка к линейной. Усложнение модели не требуется."
            return expl

        elif method == 'mediation':
            pct = result.get('percent_mediated', 0)
            indirect = result.get('indirect_effect', 0)

            if is_sig:
                expl = (f"Обнаружена медиация: {pct:.1f}% эффекта проходит через медиатор. "
                        f"Переменные связаны не напрямую, а через промежуточный механизм.")
            else:
                expl = "Медиационный эффект не обнаружен. Переменные связаны напрямую или не связаны."
            return expl

        else:
            # Базовое объяснение
            expl = f"Используя метод '{method}', получен p-value = {p_value:.4f}. "
            if is_sig:
                expl += "Это указывает на статистически значимую закономерность."
            else:
                expl += "Статистически значимой закономерности не обнаружено."
            return expl

    def _ai_interpretation(self, hypothesis: Dict, analysis_result: Dict) -> Optional[Dict]:
        """AI-интерпретация результатов."""

        # Ограничение вызовов
        self.ai_call_count += 1

        system_prompt = """Ты - опытный специалист по анализу данных и статистике. 
Объясняй результаты понятным языком для бизнес-аудитории, давай практические рекомендации.
Избегай жаргона, но будь точным."""

        # Подготовка деталей анализа
        details_str = self._format_analysis_details(analysis_result)

        user_prompt = f"""Проинтерпретируй результаты статистического анализа:

ГИПОТЕЗА: {hypothesis['text']}
ТИП АНАЛИЗА: {hypothesis['type']}
МЕТОД: {analysis_result.get('method', 'unknown')}
РЕЗУЛЬТАТЫ: {details_str}
ЗНАЧИМОСТЬ: {'Да' if analysis_result.get('is_significant') else 'Нет'}

Предоставь краткий JSON ответ:
{{
  "explanation": "2-3 предложения о том, что означает результат",
  "practical_meaning": "Практическое значение для бизнеса",
  "recommendations": ["конкретная рекомендация 1", "конкретная рекомендация 2"]
}}

Будь кратким и конкретным. Только JSON, без дополнительного текста."""

        try:
            response = self.llm_client.generate_json(user_prompt, system_prompt)

            # Проверка качества ответа
            if response and isinstance(response, dict) and 'explanation' in response:
                return response
            else:
                logger.debug("AI вернул некорректный ответ, используем базовую интерпретацию")
                return None

        except Exception as e:
            logger.debug(f"AI интерпретация недоступна: {e}")
            return None

    def _format_analysis_details(self, result: Dict) -> str:
        """Форматирование деталей для AI."""
        details = []

        if 'p_value' in result:
            details.append(f"p-value={result['p_value']:.6f}")

        if 'statistic' in result:
            details.append(f"статистика={result['statistic']:.4f}")

        if 'sample_size' in result:
            details.append(f"выборка={result['sample_size']}")

        # Специфичные для метода детали
        if 'n_clusters' in result:
            details.append(f"кластеров={result['n_clusters']}")

        if 'improvement' in result:
            details.append(f"улучшение={result['improvement']:.4f}")

        if 'percent_mediated' in result:
            details.append(f"медиация={result['percent_mediated']:.1f}%")

        return ", ".join(details)

    def _assess_practical_significance(self, result: Dict) -> str:
        """Оценка практической значимости."""
        if not result.get('is_significant'):
            return "Не значима"

        # Оценка на основе силы эффекта
        method = result.get('method')

        if method == 'segmentation':
            silhouette = result.get('silhouette_score', 0)
            if silhouette > 0.5:
                return "Высокая практическая значимость"
            elif silhouette > 0.3:
                return "Средняя практическая значимость"
            else:
                return "Низкая практическая значимость"

        elif method in ['interaction', 'nonlinear']:
            improvement = result.get('improvement', 0)
            if improvement > 0.1:
                return "Высокая практическая значимость"
            elif improvement > 0.05:
                return "Средняя практическая значимость"
            else:
                return "Низкая практическая значимость"

        elif method == 'mediation':
            pct = result.get('percent_mediated', 0)
            if pct > 50:
                return "Высокая практическая значимость"
            elif pct > 20:
                return "Средняя практическая значимость"
            else:
                return "Низкая практическая значимость"

        else:
            # Общая оценка
            effect = abs(result.get('statistic', 0))
            if effect > 0.5:
                return "Высокая практическая значимость"
            elif effect > 0.3:
                return "Средняя практическая значимость"
            else:
                return "Низкая практическая значимость"

    def _generate_recommendations(self, hypothesis: Dict, result: Dict) -> List[str]:
        """Генерация рекомендаций."""
        recommendations = []
        method = result.get('method')
        is_sig = result.get('is_significant', False)

        if is_sig:
            if method == 'segmentation':
                recommendations.append("Изучить профили сегментов для целевых стратегий")
                recommendations.append("Проверить стабильность сегментов на новых данных")

            elif method == 'interaction':
                recommendations.append("Учитывать взаимодействие при построении моделей")
                recommendations.append("Проанализировать механизм взаимодействия детальнее")

            elif method == 'threshold':
                recommendations.append("Определить точные пороговые значения")
                recommendations.append("Разработать стратегии для разных диапазонов")

            elif method == 'mediation':
                recommendations.append("Фокусироваться на медиаторе для влияния на результат")
                recommendations.append("Исследовать другие возможные медиаторы")

            else:
                recommendations.append("Исследовать причинно-следственные связи")
                recommendations.append("Проверить устойчивость результата на других данных")
        else:
            recommendations.append("Увеличить размер выборки")
            recommendations.append("Рассмотреть альтернативные гипотезы")
            recommendations.append("Проверить качество данных")

        return recommendations
