"""
Загрузка данных из различных источников.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional
import json
from dataclasses import dataclass
import logging

from ..config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Класс для представления набора данных."""
    df: pd.DataFrame
    metadata: Dict[str, Any]

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> Dict[str, Any]:
        """Извлечение метаданных из DataFrame."""
        metadata = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'numeric_columns': self._get_numeric_columns(),
            'categorical_columns': self._get_categorical_columns(),
            'datetime_columns': self._get_datetime_columns(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df)).to_dict(),
            'unique_counts': self.df.nunique().to_dict(),
            'sample_data': self._get_sample_data(),
        }
        return metadata

    def _get_numeric_columns(self) -> list:
        """Получить список числовых колонок."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols

    def _get_categorical_columns(self) -> list:
        """Получить список категориальных колонок."""
        cat_cols = self.df.select_dtypes(exclude=[np.number, 'datetime', 'datetime64']).columns.tolist()

        numeric_cols = self._get_numeric_columns()
        for col in numeric_cols:
            if self.df[col].nunique() <= 10:
                cat_cols.append(col)

        return list(set(cat_cols))

    def _get_datetime_columns(self) -> list:
        """Получить список datetime колонок."""
        datetime_cols = self.df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        return datetime_cols

    def _get_sample_data(self) -> Dict[str, list]:
        """Получить пример данных для каждой колонки."""
        sample = {}
        for col in self.df.columns:
            sample[col] = self.df[col].dropna().head(5).tolist()
        return sample

    def get_summary(self) -> str:
        """Получить текстовое описание набора данных."""
        summary = []
        summary.append(f"Набор данных: {self.metadata['shape'][0]} строк, {self.metadata['shape'][1]} колонок")
        summary.append(f"Числовые колонки: {len(self.metadata['numeric_columns'])}")
        summary.append(f"Категориальные колонки: {len(self.metadata['categorical_columns'])}")
        summary.append(f"Пропущенные значения: {self.df.isnull().sum().sum()} всего")

        return "\n".join(summary)


class DataLoader:
    """Класс для загрузки данных."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.parquet']

    def load(self, file_path: Union[str, Path]) -> Dataset:
        """
        Загрузка данных из файла.

        Args:
            file_path: Путь к файлу

        Returns:
            Объект Dataset
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == '.csv':
            df = pd.read_csv(file_path, nrows=self.config.max_rows)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, nrows=self.config.max_rows)
        elif suffix == '.json':
            df = pd.read_json(file_path)
        elif suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат: {suffix}. Поддерживаются: {self.supported_formats}")

        if len(df.columns) > self.config.max_columns:
            logger.warning(f"Слишком много колонок ({len(df.columns)}). Оставляем первые {self.config.max_columns}")
            df = df.iloc[:, :self.config.max_columns]

        dataset = Dataset(df)
        logger.info(f"Данные загружены: {dataset.get_summary()}")

        return dataset

    def load_from_dict(self, data_dict: Dict[str, list]) -> Dataset:
        """
        Загрузка данных из словаря.

        Args:
            data_dict: Словарь с данными

        Returns:
            Объект Dataset
        """
        df = pd.DataFrame(data_dict)
        dataset = Dataset(df)
        logger.info(f"Данные загружены из словаря: {dataset.get_summary()}")
        return dataset