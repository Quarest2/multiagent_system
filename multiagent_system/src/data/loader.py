"""Загрузка и предобработка данных."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from ..utils.logger import logger


class Dataset:
    """Класс для работы с датасетом."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> Dict[str, Any]:
        """Извлечение метаданных."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'missing_values': self.df.isnull().sum().to_dict()
        }

    def get_summary(self) -> str:
        """Текстовое описание датасета."""
        return (f"Датасет: {self.metadata['shape'][0]} строк, "
                f"{self.metadata['shape'][1]} колонок\n"
                f"Числовые колонки: {len(self.metadata['numeric_columns'])}\n"
                f"Категориальные: {len(self.metadata['categorical_columns'])}")


class DataLoader:
    """Загрузчик данных."""

    def load(self, filepath: str) -> Dataset:
        """Загрузка данных из файла."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")

        # Определяем формат и загружаем
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Неподдерживаемый формат: {filepath.suffix}")

        logger.info(f"Загружено {len(df)} строк, {len(df.columns)} колонок")

        # Предобработка
        df = self._preprocess(df)

        return Dataset(df)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Базовая предобработка."""
        # Удаляем полностью пустые колонки
        df = df.dropna(axis=1, how='all')

        # Удаляем константные колонки
        for col in df.columns:
            if df[col].nunique() <= 1:
                df = df.drop(columns=[col])
                logger.debug(f"Удалена константная колонка: {col}")

        # Заполняем пропуски
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown',
                               inplace=True)

        return df
