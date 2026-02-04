"""
Предобработка данных.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings

from ..config import ConfigManager
from .loader import Dataset
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PreprocessingReport:
    """Отчет о предобработке."""
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    removed_columns: List[str]
    removed_rows: int
    missing_values_imputed: int
    outliers_handled: int
    transformations_applied: List[str]
    data_quality_score: float


class DataPreprocessor:
    """Процессор для предобработки данных."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.data_config = config.get_data_config()
        self.report: Optional[PreprocessingReport] = None

    def process(self, dataset: Dataset) -> Dataset:
        """
        Основной метод предобработки данных.

        Args:
            dataset: Исходный датасет

        Returns:
            Обработанный датасет
        """
        logger.info("Начало предобработки данных")

        original_df = dataset.df.copy()
        original_shape = original_df.shape

        # Сбор информации об удаленных колонках и строках
        removed_columns = []
        transformations = []

        # 1. Удаление константных колонок
        df = self._remove_constant_columns(original_df, removed_columns)

        # 2. Удаление дубликатов
        rows_before = len(df)
        df = self._remove_duplicates(df)
        removed_duplicates = rows_before - len(df)

        # 3. Обработка пропущенных значений
        missing_before = df.isnull().sum().sum()
        df = self._handle_missing_values(df)
        missing_after = df.isnull().sum().sum()
        missing_imputed = missing_before - missing_after

        # 4. Обработка выбросов
        outliers_before = self._count_outliers(df)
        df = self._handle_outliers(df)
        outliers_after = self._count_outliers(df)
        outliers_handled = outliers_before - outliers_after

        # 5. Нормализация числовых признаков
        df, normalization_applied = self._normalize_numeric_features(df)
        if normalization_applied:
            transformations.append("normalization")

        # 6. Кодирование категориальных признаков
        df, encoding_applied = self._encode_categorical_features(df)
        if encoding_applied:
            transformations.append("encoding")

        # 7. Создание новых признаков
        df, feature_engineering_applied = self._create_new_features(df)
        if feature_engineering_applied:
            transformations.append("feature_engineering")

        # Создание отчета
        self.report = PreprocessingReport(
            original_shape=original_shape,
            processed_shape=df.shape,
            removed_columns=removed_columns,
            removed_rows=removed_duplicates,
            missing_values_imputed=missing_imputed,
            outliers_handled=outliers_handled,
            transformations_applied=transformations,
            data_quality_score=self._calculate_data_quality(df)
        )

        logger.info(f"Предобработка завершена. Форма данных: {original_shape} → {df.shape}")
        logger.info(f"Удалено колонок: {len(removed_columns)}, строк: {removed_duplicates}")
        logger.info(f"Восстановлено пропусков: {missing_imputed}")

        # Создаем новый Dataset
        processed_dataset = Dataset(df)
        processed_dataset.metadata['preprocessing_report'] = self.report.__dict__

        return processed_dataset

    def _remove_constant_columns(self, df: pd.DataFrame, removed_columns: List[str]) -> pd.DataFrame:
        """Удаление константных колонок."""
        constant_cols = []

        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            logger.info(f"Удаление константных колонок: {constant_cols}")
            removed_columns.extend(constant_cols)
            df = df.drop(columns=constant_cols)

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление дубликатов."""
        duplicates_count = df.duplicated().sum()

        if duplicates_count > 0:
            logger.info(f"Удаление {duplicates_count} дубликатов")
            df = df.drop_duplicates()

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений."""
        df_clean = df.copy()

        for col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            missing_ratio = missing_count / len(df_clean)

            if missing_count > 0:
                if missing_ratio > self.data_config.missing_threshold:
                    # Удаляем колонку, если слишком много пропусков
                    logger.warning(f"Колонка '{col}' имеет {missing_ratio:.1%} пропусков - удаление")
                    df_clean = df_clean.drop(columns=[col])
                else:
                    # Заполняем пропуски
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        # Для числовых - медиана
                        fill_value = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(fill_value)
                        logger.debug(f"Заполнение пропусков в '{col}' медианой: {fill_value}")
                    else:
                        # Для категориальных - мода
                        fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                        df_clean[col] = df_clean[col].fillna(fill_value)
                        logger.debug(f"Заполнение пропусков в '{col}' модой: {fill_value}")

        return df_clean

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка выбросов."""
        df_clean = df.copy()

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df_clean[col].nunique() > 10:  # Только для непрерывных
                # Метод межквартильного размаха
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Winsorization (ограничение вместо удаления)
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    logger.debug(f"Обработано {outliers_count} выбросов в '{col}'")

        return df_clean

    def _normalize_numeric_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Нормализация числовых признаков."""
        df_normalized = df.copy()
        applied = False

        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            # Min-Max нормализация
            for col in numeric_cols:
                if df_normalized[col].nunique() > 1:  # Не константа
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()

                    if max_val > min_val:  # Избегаем деления на 0
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                        applied = True

        return df_normalized, applied

    def _encode_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Кодирование категориальных признаков."""
        df_encoded = df.copy()
        applied = False

        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if df_encoded[col].nunique() <= 10:  # Малое количество категорий
                # One-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                applied = True
                logger.debug(f"One-hot encoding для '{col}'")
            else:
                # Frequency encoding для многих категорий
                freq = df_encoded[col].value_counts(normalize=True)
                df_encoded[col] = df_encoded[col].map(freq)
                applied = True
                logger.debug(f"Frequency encoding для '{col}'")

        return df_encoded, applied

    def _create_new_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Создание новых признаков."""
        df_with_features = df.copy()
        applied = False

        numeric_cols = df_with_features.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) >= 2:
            # Создание взаимодействий между числовыми признаками
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1:]:
                    interaction_name = f"{col1}_x_{col2}"
                    if interaction_name not in df_with_features.columns:
                        df_with_features[interaction_name] = df_with_features[col1] * df_with_features[col2]
                        applied = True
                        logger.debug(f"Создан признак взаимодействия: {interaction_name}")

            # Создание полиномиальных признаков
            for col in numeric_cols[:3]:  # Ограничиваем количество
                if col + '_squared' not in df_with_features.columns:
                    df_with_features[col + '_squared'] = df_with_features[col] ** 2
                    applied = True
                    logger.debug(f"Создан полиномиальный признак: {col}_squared")

        return df_with_features, applied

    def _count_outliers(self, df: pd.DataFrame) -> int:
        """Подсчет выбросов."""
        outlier_count = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].nunique() > 10:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_count += outliers

        return outlier_count

    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Расчет оценки качества данных."""
        if df.empty:
            return 0.0

        quality_score = 0.0
        max_score = 0.0

        # 1. Полнота данных (отсутствие пропусков)
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        quality_score += completeness * 0.3
        max_score += 0.3

        # 2. Разнообразие (не константные колонки)
        diversity_score = 0.0
        for col in df.columns:
            if df[col].nunique() > 1:
                diversity_score += 1
        diversity = diversity_score / len(df.columns) if df.columns.any() else 0
        quality_score += diversity * 0.2
        max_score += 0.2

        # 3. Баланс классов (для категориальных)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            balance_scores = []
            for col in categorical_cols:
                value_counts = df[col].value_counts(normalize=True)
                if len(value_counts) > 1:
                    # Энтропия Шеннона как мера баланса
                    entropy = stats.entropy(value_counts)
                    max_entropy = np.log(len(value_counts))
                    balance = entropy / max_entropy if max_entropy > 0 else 1
                    balance_scores.append(balance)

            if balance_scores:
                avg_balance = np.mean(balance_scores)
                quality_score += avg_balance * 0.2
                max_score += 0.2

        # 4. Распределение (нормальность для числовых)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            normality_scores = []
            for col in numeric_cols[:5]:  # Ограничиваем количество
                data = df[col].dropna()
                if len(data) > 3:
                    try:
                        # Тест на нормальность (упрощенный)
                        skewness = abs(data.skew())
                        kurtosis = abs(data.kurtosis())

                        # Чем ближе к 0, тем лучше
                        skew_score = max(0, 1 - skewness / 2)  # Считаем, что skewness < 2 приемлемо
                        kurtosis_score = max(0, 1 - abs(kurtosis - 3) / 4)  # Нормальное распределение имеет kurtosis=3

                        normality = (skew_score + kurtosis_score) / 2
                        normality_scores.append(normality)
                    except:
                        pass

            if normality_scores:
                avg_normality = np.mean(normality_scores)
                quality_score += avg_normality * 0.3
                max_score += 0.3

        # Нормализуем итоговый счет
        final_score = quality_score / max_score if max_score > 0 else 0.0

        return min(max(final_score, 0.0), 1.0)

    def get_report(self) -> Optional[PreprocessingReport]:
        """Получение отчета о предобработке."""
        return self.report

    def get_report_dict(self) -> Dict[str, Any]:
        """Получение отчета в виде словаря."""
        if self.report:
            return self.report.__dict__
        return {}