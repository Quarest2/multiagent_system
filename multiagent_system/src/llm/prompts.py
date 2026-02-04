"""
Менеджер промптов для LLM.
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path
import json

from ..config import AgentConfig


class PromptManager:
    """Менеджер промптов."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Загрузка промптов из файла."""
        default_prompts = self._get_default_prompts()

        prompts_path = Path("config/llm_prompts.yaml")
        if prompts_path.exists():
            try:
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    file_prompts = yaml.safe_load(f)

                default_prompts.update(file_prompts)
            except Exception as e:
                print(f"Ошибка загрузки промптов из файла: {e}")

        return default_prompts

    def _get_default_prompts(self) -> Dict[str, str]:
        """Промпты по умолчанию."""
        return {
            'hypothesis_generation': """
            Ты - опытный аналитик данных. Проанализируй информацию о наборе данных и предложи исследовательские гипотезы.

            Информация о данных:
            {dataset_summary}

            Примеры значений колонок:
            {column_info}

            Числовые колонки: {numeric_columns}
            Категориальные колонки: {categorical_columns}

            Сформулируй 5-10 содержательных гипотез для проверки. Для каждой гипотезы укажи:
            1. Формулировку гипотезы
            2. Тип гипотезы (сравнение, корреляция, распределение и т.д.)
            3. Какие колонки задействованы
            4. Ожидаемую значимость (высокая, средняя, низкая)

            Гипотезы должны быть проверяемыми статистическими методами.
            """,

            'qa_inspection': """
            Ты - эксперт по статистике и анализу данных. Проверь качество гипотезы и метода анализа.

            Гипотеза: {hypothesis_text}
            Тип гипотезы: {hypothesis_type}
            Метод анализа: {method}

            Результаты анализа:
            P-value: {p_value}
            Статистика: {statistic}

            Информация о данных:
            {dataset_summary}

            Колонки, задействованные в гипотезе: {columns_involved}

            Проверь:
            1. Корректность формулировки гипотезы
            2. Соответствие метода типу гипотезы и данным
            3. Корректность статистических предположений
            4. Возможные угрозы валидности
            5. Альтернативные объяснения результатов

            Выяви проблемы и дай рекомендации по улучшению.
            Если нужна доработка гипотезы или метода, укажи это явно.
            """,

            'hypothesis_refinement': """
            Ты помогаешь улучшить исследовательскую гипотезу на основе обратной связи.

            Исходная гипотеза: {original_hypothesis}
            Тип гипотезы: {hypothesis_type}
            Колонки: {columns_involved}

            Обратная связь:
            {feedback}

            Цель доработки: {refinement_goal}

            Предложи улучшенную формулировку гипотезы, которая:
            1. Учитывает замечания из обратной связи
            2. Более точна и проверяема
            3. Учитывает особенности данных
            4. Имеет четкие условия и границы применимости

            Верни только улучшенную формулировку гипотезы.
            """,

            'interpretation': """
            Ты помогаешь интерпретировать результаты статистического анализа.

            Гипотеза: {hypothesis_text}
            Метод: {method}
            P-value: {p_value}
            Статистика: {statistic}
            Размер эффекта: {effect_size}
            Доверительный интервал: {confidence_interval}

            Дай содержательную интерпретацию результатов:
            1. Основной вывод (подтверждена/отклонена гипотеза)
            2. Практическая значимость
            3. Ограничения анализа
            4. Рекомендации для дальнейшего исследования

            Будь точным, но понятным для неспециалистов.
            """,

            'report_generation': """
            Создай аналитический отчет на основе результатов проверки гипотез.

            Всего гипотез: {total_hypotheses}
            Значимых гипотез: {significant_hypotheses}

            Ключевые находки:
            {key_findings}

            Структура отчета:
            1. Краткое резюме
            2. Методология
            3. Основные результаты
            4. Выводы и рекомендации
            5. Ограничения исследования

            Сделай отчет профессиональным, но доступным.
            """
        }

    def get_prompt(self,
                   prompt_name: str,
                   **kwargs) -> str:
        """
        Получение промпта с подстановкой значений.

        Args:
            prompt_name: Название промпта
            **kwargs: Значения для подстановки

        Returns:
            Заполненный промпт
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Промпт '{prompt_name}' не найден")

        prompt = self.prompts[prompt_name]

        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                if isinstance(value, (list, dict)):
                    value_str = json.dumps(value, ensure_ascii=False, indent=2)
                else:
                    value_str = str(value)

                prompt = prompt.replace(placeholder, value_str)

        return prompt

    def save_prompts(self, path: str):
        """
        Сохранение промптов в файл.

        Args:
            path: Путь для сохранения
        """
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.prompts, f, allow_unicode=True, default_flow_style=False)