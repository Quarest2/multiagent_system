"""
Валидация данных и гипотез.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from scipy import stats
import warnings

from ..config import ConfigManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ValidationResult:
    """Результат валидации."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    confidence_score: float


class DataValidator:
    """Валидатор данных и гипотез."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.agent_config = config.get_agent_config()

    def validate_dataset(self, df: pd.DataFrame) -> ValidationResult:
        """
        Валидация датасета.

        Args:
            df: Датафрейм для валидации

        Returns:
            Результат валидации
        """
        errors = []
        warnings = []
        suggestions = []

        # Базовые проверки
        if df.empty:
            errors.append("Датасет пустой")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                confidence_score=0.0
            )

        # 1. Проверка размера
        if len(df) < 10:
            warnings.append(f"Маленький размер выборки: {len(df)} строк")
            suggestions.append("Рассмотрите увеличение размера выборки")

        if len(df.columns) < 2:
            errors.append(f"Слишком мало колонок: {len(df.columns)}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                confidence_score=0.0
            )

        # 2. Проверка пропущенных значений
        missing_total = df.isnull().sum().sum()
        missing_ratio = missing_total / (len(df) * len(df.columns))

        if missing_ratio > self.agent_config.significance_level:
            warnings.append(f"Много пропущенных значений: {missing_ratio:.1%}")
            suggestions.append("Обработайте пропущенные значения перед анализом")

        # 3. Проверка типов данных
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) == 0 and len(categorical_cols) == 0:
            errors.append("Нет подходящих колонок для анализа")

        # 4. Проверка уникальности
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Обнаружено {duplicate_count} дубликатов")
            suggestions.append("Удалите дубликаты для улучшения качества анализа")

        # 5. Проверка выбросов (для числовых колонок)
        outlier_warnings = self._check_outliers(df, numeric_cols)
        warnings.extend(outlier_warnings)

        # Вычисляем оценку уверенности
        confidence = self._calculate_dataset_confidence(
            df, len(errors), len(warnings), missing_ratio
        )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            confidence_score=confidence
        )

    def validate_hypothesis(self, hypothesis: Dict[str, Any], df: pd.DataFrame) -> ValidationResult:
        """
        Валидация гипотезы.

        Args:
            hypothesis: Словарь с гипотезой
            df: Датафрейм с данными

        Returns:
            Результат валидации
        """
        errors = []
        warnings = []
        suggestions = []

        hypothesis_text = hypothesis.get('text', '')
        hypothesis_type = hypothesis.get('type', '')
        columns = hypothesis.get('columns', [])

        # 1. Проверка наличия обязательных полей
        if not hypothesis_text:
            errors.append("Гипотеза не имеет текстового описания")

        if not hypothesis_type:
            warnings.append("Тип гипотезы не указан")

        if not columns:
            errors.append("Гипотеза не ссылается на колонки данных")

        # 2. Проверка существования колонок
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Колонки не найдены в данных: {missing_columns}")

        # 3. Проверка совместимости типов данных
        if columns and len(missing_columns) == 0:
            type_errors = self._check_column_compatibility(columns, hypothesis_type, df)
            errors.extend(type_errors)

        # 4. Проверка статистической проверяемости
        testability_warnings = self._check_testability(hypothesis_type, columns, df)
        warnings.extend(testability_warnings)

        # 5. Проверка практической значимости
        significance_suggestions = self._check_practical_significance(hypothesis, df)
        suggestions.extend(significance_suggestions)

        # Вычисляем оценку уверенности
        confidence = self._calculate_hypothesis_confidence(
            hypothesis, df, len(errors), len(warnings)
        )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            confidence_score=confidence
        )

    def validate_statistical_test(self, test_name: str, data_requirements: Dict[str, Any]) -> ValidationResult:
        """
        Валидация статистического теста.

        Args:
            test_name: Название теста
            data_requirements: Требования к данным

        Returns:
            Результат валидации
        """
        errors = []
        warnings = []
        suggestions = []

        # Требования для разных тестов
        test_requirements = {
            't_test_ind': ['numeric', 'categorical_2groups', 'min_sample_size_30'],
            't_test_rel': ['numeric_paired', 'min_sample_size_30'],
            'anova': ['numeric', 'categorical_3+groups', 'min_sample_size_per_group_10'],
            'pearson': ['numeric', 'linear_relationship', 'normality'],
            'spearman': ['numeric', 'monotonic_relationship'],
            'chi2': ['categorical', 'min_expected_frequency_5'],
            'mann_whitney': ['numeric', 'categorical_2groups', 'non_normal'],
            'kruskal_wallis': ['numeric', 'categorical_3+groups', 'non_normal']
        }

        if test_name not in test_requirements:
            warnings.append(f"Неизвестный тест: {test_name}")
            suggestions.append("Проверьте корректность названия теста")

        # Проверяем соответствие требованиям
        if test_name in test_requirements:
            requirements = test_requirements[test_name]
            missing_requirements = []

            for req in requirements:
                if req not in data_requirements.get('met_requirements', []):
                    missing_requirements.append(req)

            if missing_requirements:
                warnings.append(f"Тест {test_name} требует: {missing_requirements}")
                suggestions.append(f"Убедитесь, что данные соответствуют требованиям теста")

        # Проверяем размер выборки
        sample_size = data_requirements.get('sample_size', 0)
        if sample_size < 30:
            warnings.append(f"Маленький размер выборки для теста: {sample_size}")
            suggestions.append("Рассмотрите использование непараметрических тестов")

        # Вычисляем оценку уверенности
        confidence = self._calculate_test_confidence(
            test_name, data_requirements, len(errors), len(warnings)
        )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            confidence_score=confidence
        )

    def _check_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """Проверка выбросов."""
        warnings = []

        for col in numeric_cols[:5]:  # Проверяем только первые 5 числовых колонок
            if df[col].nunique() > 10:  # Только для непрерывных
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_ratio = outliers / len(df)

                if outlier_ratio > 0.05:  # Более 5% выбросов
                    warnings.append(f"Колонка '{col}' имеет {outlier_ratio:.1%} выбросов")

        return warnings

    def _check_column_compatibility(self, columns: List[str], hypothesis_type: str, df: pd.DataFrame) -> List[str]:
        """Проверка совместимости типов колонок с типом гипотезы."""
        errors = []

        if hypothesis_type == 'correlation':
            # Для корреляции нужны две числовые колонки
            if len(columns) != 2:
                errors.append(f"Корреляция требует ровно 2 колонки, получено {len(columns)}")
            else:
                col1, col2 = columns[0], columns[1]
                if not (pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2])):
                    errors.append(f"Для корреляции нужны числовые колонки: {col1}, {col2}")

        elif hypothesis_type == 'mean_difference':
            # Для сравнения средних нужна числовая и категориальная колонки
            if len(columns) != 2:
                errors.append(f"Сравнение средних требует ровно 2 колонки, получено {len(columns)}")
            else:
                col1, col2 = columns[0], columns[1]
                is_numeric_col1 = pd.api.types.is_numeric_dtype(df[col1])
                is_numeric_col2 = pd.api.types.is_numeric_dtype(df[col2])

                if not (is_numeric_col1 ^ is_numeric_col2):  # XOR - ровно одна должна быть числовой
                    errors.append(
                        f"Сравнение средних требует одну числовую и одну категориальную колонку: {col1}, {col2}")

                # Проверяем количество категорий
                categorical_col = col2 if is_numeric_col1 else col1
                unique_categories = df[categorical_col].nunique()
                if unique_categories < 2:
                    errors.append(f"Категориальная колонка '{categorical_col}' должна иметь минимум 2 категории")

        elif hypothesis_type == 'distribution':
            # Для проверки распределения нужна одна числовая колонка
            if len(columns) != 1:
                errors.append(f"Проверка распределения требует 1 колонку, получено {len(columns)}")
            else:
                col = columns[0]
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Для проверки распределения нужна числовая колонка: {col}")

        return errors

    def _check_testability(self, hypothesis_type: str, columns: List[str], df: pd.DataFrame) -> List[str]:
        """Проверка проверяемости гипотезы."""
        warnings = []

        if hypothesis_type == 'mean_difference' and len(columns) == 2:
            # Определяем, какая колонка категориальная
            col1, col2 = columns[0], columns[1]
            categorical_col = col1 if not pd.api.types.is_numeric_dtype(df[col1]) else col2

            # Проверяем размер групп
            group_sizes = df[categorical_col].value_counts()
            min_group_size = group_sizes.min()

            if min_group_size < 10:
                warnings.append(f"Маленький размер группы в '{categorical_col}': {min_group_size}")
            elif min_group_size < 30:
                warnings.append(f"Небольшой размер группы в '{categorical_col}': {min_group_size}")

        elif hypothesis_type == 'correlation' and len(columns) == 2:
            col1, col2 = columns[0], columns[1]

            # Проверяем линейность (упрощенно)
            correlation = df[[col1, col2]].corr().iloc[0, 1]
            if abs(correlation) < 0.1:
                warnings.append(f"Слабая корреляция между '{col1}' и '{col2}': {correlation:.3f}")

        return warnings

    def _check_practical_significance(self, hypothesis: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Проверка практической значимости."""
        suggestions = []
        columns = hypothesis.get('columns', [])

        if hypothesis.get('type') == 'mean_difference' and len(columns) == 2:
            # Для сравнения средних
            col1, col2 = columns[0], columns[1]
            numeric_col = col1 if pd.api.types.is_numeric_dtype(df[col1]) else col2
            categorical_col = col2 if numeric_col == col1 else col1

            # Рассчитываем размер эффекта (Cohen's d)
            groups = df[categorical_col].unique()
            if len(groups) == 2:
                group1 = df[df[categorical_col] == groups[0]][numeric_col]
                group2 = df[df[categorical_col] == groups[1]][numeric_col]

                if len(group1) > 0 and len(group2) > 0:
                    mean_diff = group1.mean() - group2.mean()
                    pooled_std = np.sqrt((group1.std() ** 2 + group2.std() ** 2) / 2)

                    if pooled_std > 0:
                        cohens_d = abs(mean_diff) / pooled_std

                        if cohens_d < 0.2:
                            suggestions.append(f"Маленький размер эффекта (Cohen's d = {cohens_d:.2f})")
                        elif cohens_d < 0.5:
                            suggestions.append(f"Средний размер эффекта (Cohen's d = {cohens_d:.2f})")
                        else:
                            suggestions.append(f"Большой размер эффекта (Cohen's d = {cohens_d:.2f})")

        elif hypothesis.get('type') == 'correlation' and len(columns) == 2:
            # Для корреляции
            col1, col2 = columns[0], columns[1]
            correlation = df[[col1, col2]].corr().iloc[0, 1]

            if abs(correlation) < 0.3:
                suggestions.append(f"Слабая корреляция (r = {correlation:.2f}) - может быть статистически незначима")
            elif abs(correlation) < 0.5:
                suggestions.append(f"Умеренная корреляция (r = {correlation:.2f})")
            else:
                suggestions.append(f"Сильная корреляция (r = {correlation:.2f})")

        return suggestions

    def _calculate_dataset_confidence(self, df: pd.DataFrame, error_count: int,
                                      warning_count: int, missing_ratio: float) -> float:
        """Вычисление уверенности в качестве датасета."""
        base_confidence = 1.0

        # Штрафы за ошибки
        base_confidence -= error_count * 0.3

        # Штрафы за предупреждения
        base_confidence -= warning_count * 0.1

        # Штраф за пропущенные значения
        base_confidence -= missing_ratio * 0.5

        # Бонус за размер выборки
        if len(df) > 100:
            base_confidence += 0.1
        elif len(df) > 30:
            base_confidence += 0.05

        # Бонус за разнообразие колонок
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            base_confidence += 0.1

        return max(0.0, min(base_confidence, 1.0))

    def _calculate_hypothesis_confidence(self, hypothesis: Dict[str, Any], df: pd.DataFrame,
                                         error_count: int, warning_count: int) -> float:
        """Вычисление уверенности в гипотезе."""
        base_confidence = 0.7  # Базовая уверенность

        # Штрафы за ошибки
        base_confidence -= error_count * 0.3

        # Штрафы за предупреждения
        base_confidence -= warning_count * 0.1

        # Бонус за конкретность гипотезы
        hypothesis_text = hypothesis.get('text', '')
        if len(hypothesis_text.split()) > 8:  # Подробная формулировка
            base_confidence += 0.1

        # Бонус за наличие обоснования
        if hypothesis.get('reasoning'):
            base_confidence += 0.1

        # Бонус за правильный тип гипотезы
        valid_types = ['correlation', 'mean_difference', 'distribution', 'association', 'trend']
        if hypothesis.get('type') in valid_types:
            base_confidence += 0.1

        return max(0.0, min(base_confidence, 1.0))

    def _calculate_test_confidence(self, test_name: str, data_requirements: Dict[str, Any],
                                   error_count: int, warning_count: int) -> float:
        """Вычисление уверенности в тесте."""
        base_confidence = 0.8

        # Штрафы за ошибки
        base_confidence -= error_count * 0.4

        # Штрафы за предупреждения
        base_confidence -= warning_count * 0.15

        # Бонус за общепринятые тесты
        common_tests = ['t_test_ind', 'anova', 'pearson', 'chi2', 'mann_whitney']
        if test_name in common_tests:
            base_confidence += 0.1

        # Бонус за достаточный размер выборки
        sample_size = data_requirements.get('sample_size', 0)
        if sample_size > 100:
            base_confidence += 0.1
        elif sample_size > 30:
            base_confidence += 0.05

        return max(0.0, min(base_confidence, 1.0))

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Получение сводки по данным."""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'missing_values': df.isnull().sum().sum(),
            'missing_ratio': df.isnull().sum().sum() / (len(df) * len(df.columns)) if len(df) > 0 else 0,
            'duplicates': df.duplicated().sum(),
            'basic_stats': self._get_basic_stats(df)
        }

    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Получение базовой статистики."""
        stats = {}

        # Для числовых колонок
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe().to_dict()
            stats['numeric'] = numeric_stats

        # Для категориальных колонок
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_stats = {}
            for col in categorical_cols[:10]:  # Ограничиваем количество
                cat_stats[col] = {
                    'unique_count': df[col].nunique(),
                    'mode': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'mode_percentage': (df[col].value_counts(normalize=True).iloc[0] * 100
                                        if len(df[col].value_counts()) > 0 else 0)
                }
            stats['categorical'] = cat_stats

        return stats