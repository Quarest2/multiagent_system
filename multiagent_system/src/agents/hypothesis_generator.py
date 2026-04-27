"""Агент генерации гипотез с AI."""

import itertools
import numpy as np
from typing import List, Dict, Any
from ..config import AgentConfig, LLMConfig
from ..data.loader import Dataset
from ..llm.groq_client import GroqLLMClient
from ..utils.logger import logger


class HypothesisGenerator:
    """Генератор статистических гипотез с AI."""
    
    def __init__(self, config: AgentConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_client = GroqLLMClient(llm_config.api_key) if llm_config.enabled else None
        self.hypothesis_id = 0

    def _identify_target_variable(self, dataset: Dataset) -> Optional[str]:
        """Автоопределение целевой переменной."""
        keywords = ['left', 'churn', 'attrition', 'converted', 'purchased',
                    'churned', 'retained', 'outcome', 'target', 'label']

        for col in dataset.df.columns:
            if any(kw in col.lower() for kw in keywords):
                if dataset.df[col].nunique() == 2:  # Бинарная
                    logger.info(f"🎯 Целевая переменная: {col}")
                    return col
        return None

    def generate(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Генерация гипотез."""
        hypotheses = []
        
        # Статистические гипотезы (базовые)
        stat_hypotheses = self._generate_statistical_hypotheses(dataset)
        hypotheses.extend(stat_hypotheses)
        
        # AI-генерация гипотез (если доступно)
        if self.config.use_ai_generation and self.llm_client and self.llm_client.is_available():
            ai_hypotheses = self._generate_ai_hypotheses(dataset)
            if ai_hypotheses:
                hypotheses.extend(ai_hypotheses)
                logger.info(f"AI сгенерировал {len(ai_hypotheses)} дополнительных гипотез")
        
        # Убираем дубликаты и ограничиваем количество
        hypotheses = self._deduplicate(hypotheses)
        hypotheses.sort(key=lambda x: x['priority'])
        hypotheses = hypotheses[:self.config.max_hypotheses]
        
        logger.info(f"Итого сгенерировано {len(hypotheses)} гипотез")
        return hypotheses

    def _generate_statistical_hypotheses(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Улучшенная статистическая генерация."""
        hypotheses = []
        numeric_cols = dataset.metadata['numeric_columns']
        categorical_cols = dataset.metadata['categorical_columns']

        # ШАГ 1: ПРИОРИТЕТ - Целевая переменная
        target_var = self._identify_target_variable(dataset)
        if target_var:
            hypotheses.extend(self._generate_target_hypotheses(target_var, dataset))

        # ШАГ 2: Сегментация (кластеры)
        if len(numeric_cols) >= 2:
            # Фильтрация: только числовые
            clustering_cols = [c for c in numeric_cols if c in dataset.df.select_dtypes(include=[np.number]).columns][
                :4]

            if len(clustering_cols) >= 2:
                hypotheses.append({
                    'id': self._next_id(),
                    'text': f"Данные содержат скрытые сегменты по переменным {clustering_cols}",
                    'type': 'segmentation',
                    'columns': clustering_cols,
                    'priority': 1,
                    'source': 'statistical'
                })

        # ШАГ 3: Взаимодействия (топ-5)
        if len(numeric_cols) >= 3:
            for i, target in enumerate(numeric_cols[:2]):
                predictors = [c for c in numeric_cols if c != target][:2]
                hypotheses.append({
                    'id': self._next_id(),
                    'text': f"Влияние на '{target}' зависит от взаимодействия переменных",
                    'type': 'interaction_effect',
                    'columns': [target] + predictors,
                    'priority': 2,
                    'source': 'statistical'
                })

        # ШАГ 4: Пороговые эффекты
        for num_col in numeric_cols[:2]:
            hypotheses.append({
                'id': self._next_id(),
                'text': f"Существует пороговое значение для '{num_col}'",
                'type': 'threshold_effect',
                'columns': [num_col],
                'priority': 2,
                'source': 'statistical'
            })

        # ШАГ 5: Нелинейные зависимости (только для сильных корреляций)
        strong_pairs = self._find_strong_correlations(dataset, threshold=0.3)
        for col1, col2 in strong_pairs[:2]:
            hypotheses.append({
                'id': self._next_id(),
                'text': f"Нелинейная (U-образная) зависимость между '{col1}' и '{col2}'",
                'type': 'nonlinear_relationship',
                'columns': [col1, col2],
                'priority': 2,
                'source': 'statistical'
            })

        # ШАГ 6: Медиация (если есть целевая)
        if target_var and len(numeric_cols) >= 2:
            mediator = numeric_cols[0] if numeric_cols[0] != target_var else numeric_cols[1]
            predictor = numeric_cols[1] if numeric_cols[1] != target_var and numeric_cols[1] != mediator else \
            numeric_cols[2] if len(numeric_cols) > 2 else None

            if predictor:
                hypotheses.append({
                    'id': self._next_id(),
                    'text': f"'{predictor}' влияет на '{target_var}' через '{mediator}'",
                    'type': 'mediation_analysis',
                    'columns': [predictor, mediator, target_var],
                    'priority': 3,
                    'source': 'statistical'
                })

        # ШАГ 7: ТОЛЬКО ЕСЛИ ОСТАЛОСЬ МЕСТО - простые корреляции
        remaining_slots = max(0, 15 - len(hypotheses))  # Резервируем место для AI
        if remaining_slots > 0:
            for col1, col2 in itertools.combinations(numeric_cols, 2):
                if remaining_slots <= 0:
                    break
                hypotheses.append({
                    'id': self._next_id(),
                    'text': f"Корреляция между '{col1}' и '{col2}'",
                    'type': 'correlation',
                    'columns': [col1, col2],
                    'priority': 5,  # Низкий приоритет
                    'source': 'statistical'
                })
                remaining_slots -= 1

        # Сравнение средних (низкий приоритет)
        for num_col in numeric_cols[:2]:
            for cat_col in categorical_cols[:2]:
                if dataset.df[cat_col].nunique() >= 2:
                    hypotheses.append({
                        'id': self._next_id(),
                        'text': f"Среднее '{num_col}' различается по '{cat_col}'",
                        'type': 'mean_difference',
                        'columns': [num_col, cat_col],
                        'priority': 4,
                        'source': 'statistical'
                    })

        return hypotheses

    def _generate_target_hypotheses(self, target: str, dataset: Dataset) -> List[Dict]:
        """Гипотезы для целевой переменной."""
        hypotheses = []
        numeric_cols = [c for c in dataset.metadata['numeric_columns'] if c != target]
        categorical_cols = dataset.metadata['categorical_columns']

        # Сегментация по целевой
        if len(numeric_cols) >= 2:
            hypotheses.append({
                'id': self._next_id(),
                'text': f"Записи с {target}=1 и {target}=0 образуют разные сегменты",
                'type': 'segmentation',
                'columns': numeric_cols[:4] + [target],
                'priority': 1,
                'source': 'statistical',
                'focus': 'target'
            })

        # Предиктивная регрессия
        if len(numeric_cols) >= 1:
            hypotheses.append({
                'id': self._next_id(),
                'text': f"Можно предсказать '{target}' по другим переменным",
                'type': 'regression',
                'columns': [target] + numeric_cols[:3],
                'priority': 1,
                'source': 'statistical'
            })

        return hypotheses

    # ДОБАВИТЬ вспомогательный метод:

    def _find_strong_correlations(self, dataset: Dataset, threshold: float = 0.3) -> List[tuple]:
        """Поиск сильных корреляций."""
        numeric_cols = dataset.metadata['numeric_columns']
        strong_pairs = []

        for col1, col2 in itertools.combinations(numeric_cols, 2):
            data = dataset.df[[col1, col2]].dropna()
            if len(data) < 10:
                continue

            from scipy.stats import pearsonr
            corr, _ = pearsonr(data[col1], data[col2])

            if abs(corr) >= threshold:
                strong_pairs.append((col1, col2))

        return strong_pairs
    
    def _generate_ai_hypotheses(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """AI-генерация гипотез."""
        system_prompt = """Ты - эксперт по анализу данных и статистике.
Твоя задача - генерировать интересные и проверяемые исследовательские гипотезы."""
        
        # Подготовка данных о датасете
        data_summary = self._prepare_dataset_summary(dataset)
        
        user_prompt = f"""Проанализируй датасет и предложи 3-5 содержательных статистических гипотез.

ИНФОРМАЦИЯ О ДАННЫХ:
{data_summary}

Верни ответ в JSON формате:
{{
  "hypotheses": [
    {{
      "text": "формулировка гипотезы",
      "type": "correlation|mean_difference|regression|clustering",
      "columns": ["col1", "col2"],
      "reasoning": "почему эта гипотеза интересна"
    }}
  ]
}}

Требования:
- Гипотезы должны быть проверяемыми статистическими методами
- Используй только существующие колонки
- Формулируй понятно и конкретно
"""
        
        response = self.llm_client.generate_json(user_prompt, system_prompt)
        
        if not response or 'hypotheses' not in response:
            return []
        
        ai_hypotheses = []
        for h in response['hypotheses']:
            ai_hypotheses.append({
                'id': self._next_id(),
                'text': h.get('text', ''),
                'type': h.get('type', 'correlation'),
                'columns': h.get('columns', []),
                'priority': 1,
                'source': 'ai',
                'reasoning': h.get('reasoning', '')
            })
        
        return ai_hypotheses
    
    def _prepare_dataset_summary(self, dataset: Dataset) -> str:
        """Подготовка описания датасета для AI."""
        summary_parts = []
        
        summary_parts.append(f"Размер: {dataset.metadata['shape'][0]} строк, {dataset.metadata['shape'][1]} колонок")
        summary_parts.append(f"\nЧисловые колонки ({len(dataset.metadata['numeric_columns'])}):")
        for col in dataset.metadata['numeric_columns']:
            stats = dataset.df[col].describe()
            summary_parts.append(f"  - {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                                f"min={stats['min']:.2f}, max={stats['max']:.2f}")
        
        summary_parts.append(f"\nКатегориальные колонки ({len(dataset.metadata['categorical_columns'])}):")
        for col in dataset.metadata['categorical_columns']:
            unique_vals = dataset.df[col].unique()[:5]
            summary_parts.append(f"  - {col}: {len(dataset.df[col].unique())} уникальных значений "
                                f"({', '.join(map(str, unique_vals))}...)")
        
        return '\n'.join(summary_parts)
    
    def _deduplicate(self, hypotheses: List[Dict]) -> List[Dict]:
        """Удаление дубликатов."""
        seen = set()
        unique = []
        
        for h in hypotheses:
            key = (h['type'], tuple(sorted(h['columns'])))
            if key not in seen:
                seen.add(key)
                unique.append(h)
        
        return unique
    
    def _next_id(self) -> int:
        self.hypothesis_id += 1
        return self.hypothesis_id
