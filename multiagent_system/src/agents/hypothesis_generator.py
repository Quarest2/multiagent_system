"""
Генератор гипотез с использованием LLM и статистических эвристик.
"""

import itertools
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

from ..config import AgentConfig
from ..data.loader import Dataset
from ..llm.base_client import LLMClient
from ..llm.prompts import PromptManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Hypothesis:
    """Класс гипотезы."""
    id: int
    text: str
    type: str
    columns: List[str]
    priority: int = 1
    llm_generated: bool = False
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'text': self.text,
            'type': self.type,
            'columns': self.columns,
            'priority': self.priority,
            'llm_generated': self.llm_generated,
            'metadata': self.metadata or {}
        }


class HypothesisGenerator:
    """Генератор гипотез."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_client = LLMClient(config) if config.enable_llm else None
        self.prompt_manager = PromptManager(config)
        self.hypothesis_id = 0

    def generate(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """
        Генерация гипотез для набора данных.

        Args:
            dataset: Объект Dataset

        Returns:
            Список гипотез в виде словарей
        """
        logger.info("Генерация гипотез...")

        hypotheses = []

        statistical_hypotheses = self._generate_statistical_hypotheses(dataset)
        hypotheses.extend(statistical_hypotheses)

        if self.config.enable_llm and self.llm_client:
            llm_hypotheses = self._generate_llm_hypotheses(dataset)
            hypotheses.extend(llm_hypotheses)

        hypotheses.sort(key=lambda x: x['priority'])

        if self.config.max_hypotheses and len(hypotheses) > self.config.max_hypotheses:
            hypotheses = hypotheses[:self.config.max_hypotheses]

        logger.info(f"Сгенерировано {len(hypotheses)} гипотез")
        return hypotheses

    def _generate_statistical_hypotheses(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Генерация статистических гипотез на основе эвристик."""
        hypotheses = []
        df = dataset.df
        metadata = dataset.metadata

        numeric_cols = metadata['numeric_columns']
        categorical_cols = metadata['categorical_columns']

        if len(numeric_cols) >= 2:
            for col1, col2 in itertools.combinations(numeric_cols, 2):
                hypothesis = Hypothesis(
                    id=self._next_id(),
                    text=f"Существует корреляция между '{col1}' и '{col2}'",
                    type='correlation',
                    columns=[col1, col2],
                    priority=1,
                    llm_generated=False,
                    metadata={'heuristic': 'numeric_correlation'}
                )
                hypotheses.append(hypothesis.to_dict())

        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                if cat_col in df.columns and df[cat_col].nunique() >= 2:
                    hypothesis = Hypothesis(
                        id=self._next_id(),
                        text=f"Среднее значение '{num_col}' различается между группами в '{cat_col}'",
                        type='mean_difference',
                        columns=[num_col, cat_col],
                        priority=1,
                        llm_generated=False,
                        metadata={'heuristic': 'mean_difference'}
                    )
                    hypotheses.append(hypothesis.to_dict())

        for num_col in numeric_cols[:3]:
            hypothesis = Hypothesis(
                id=self._next_id(),
                text=f"Распределение '{num_col}' отличается от нормального",
                type='normality',
                columns=[num_col],
                priority=2,
                llm_generated=False,
                metadata={'heuristic': 'normality_test'}
            )
            hypotheses.append(hypothesis.to_dict())

        if len(categorical_cols) >= 2:
            for col1, col2 in itertools.combinations(categorical_cols[:4], 2):
                hypothesis = Hypothesis(
                    id=self._next_id(),
                    text=f"Переменные '{col1}' и '{col2}' независимы",
                    type='independence',
                    columns=[col1, col2],
                    priority=2,
                    llm_generated=False,
                    metadata={'heuristic': 'categorical_independence'}
                )
                hypotheses.append(hypothesis.to_dict())

        return hypotheses

    def _generate_llm_hypotheses(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Генерация гипотез с помощью LLM."""
        try:
            prompt = self.prompt_manager.get_prompt(
                'hypothesis_generation',
                dataset_summary=dataset.get_summary(),
                column_info=str(dataset.metadata['sample_data']),
                numeric_columns=dataset.metadata['numeric_columns'],
                categorical_columns=dataset.metadata['categorical_columns']
            )

            response = self.llm_client.generate(prompt, max_tokens=1000)

            llm_hypotheses = self._parse_llm_response(response, dataset)

            logger.info(f"LLM сгенерировал {len(llm_hypotheses)} гипотез")
            return llm_hypotheses

        except Exception as e:
            logger.error(f"Ошибка генерации гипотез через LLM: {e}")
            return []

    def _parse_llm_response(self, response: str, dataset: Dataset) -> List[Dict[str, Any]]:
        """Парсинг ответа LLM."""
        hypotheses = []

        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line or len(line) < 20:
                continue

            columns_found = []
            for col in dataset.df.columns:
                if col in line:
                    columns_found.append(col)

            if columns_found:
                hypothesis = Hypothesis(
                    id=self._next_id(),
                    text=line,
                    type='llm_generated',
                    columns=columns_found,
                    priority=2,
                    llm_generated=True,
                    metadata={'source': 'llm', 'parsed_from': line[:50]}
                )
                hypotheses.append(hypothesis.to_dict())

        return hypotheses

    def _next_id(self) -> int:
        """Получить следующий ID гипотезы."""
        self.hypothesis_id += 1
        return self.hypothesis_id