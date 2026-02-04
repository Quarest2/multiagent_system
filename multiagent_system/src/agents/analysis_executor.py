"""
Исполнитель статистического анализа.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional

from ..config import AgentConfig
from ..data.loader import Dataset
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class AnalysisExecutor:
    """Исполнитель статистического анализа."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.alpha = config.significance_level

    def execute(self,
                hypothesis: Dict[str, Any],
                method: str,
                dataset: Dataset) -> Dict[str, Any]:
        """
        Выполнение статистического анализа.

        Args:
            hypothesis: Гипотеза
            method: Метод анализа
            dataset: Набор данных

        Returns:
            Результаты анализа
        """
        try:
            if method in self._get_methods():
                return self._get_methods()[method](hypothesis, dataset)
            else:
                return self._unknown_method(hypothesis, method)

        except Exception as e:
            logger.error(f"Ошибка выполнения анализа: {e}")
            return self._error_result(hypothesis, method, str(e))

    def _get_methods(self):
        """Получение словаря методов."""
        return {
            'pearson': self._pearson_correlation,
            'spearman': self._spearman_correlation,
            'kendall': self._kendall_correlation,
            't_test_ind': self._t_test_independent,
            't_test_rel': self._t_test_related,
            'mannwhitneyu': self._mannwhitneyu,
            'wilcoxon': self._wilcoxon,
            'anova': self._anova,
            'kruskal': self._kruskal,
            'chi2': self._chi_square,
            'fisher_exact': self._fisher_exact,
            'shapiro': self._shapiro_wilk,
            'normaltest': self._normaltest,
            'anderson': self._anderson_darling,
            'levene': self._levene,
            'bartlett': self._bartlett
        }

    def _pearson_correlation(self, hypothesis: Dict, dataset: Dataset) -> Dict[str, Any]:
        """Корреляция Пирсона."""
        columns = self._get_columns(hypothesis, 2)
        col1, col2 = columns

        df = dataset.df.dropna(subset=[col1, col2])

        if len(df) < 3:
            raise ValueError(f"Недостаточно данных для корреляции: {len(df)} наблюдений")

        corr, p_value = stats.pearsonr(df[col1], df[col2])

        return {
            'statistic': corr,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'effect_size': abs(corr),
            'confidence_interval': self._bootstrap_ci(df[col1], df[col2], stats.pearsonr),
            'n': len(df)
        }

    def _t_test_independent(self, hypothesis: Dict, dataset: Dataset) -> Dict[str, Any]:
        """t-тест для независимых выборок."""
        columns = self._get_columns(hypothesis, 2)
        num_col, cat_col = columns

        df = dataset.df.dropna(subset=[num_col, cat_col])

        groups = df[cat_col].unique()
        if len(groups) != 2:
            raise ValueError(f"Требуется ровно 2 группы, найдено: {len(groups)}")

        group1 = df[df[cat_col] == groups[0]][num_col]
        group2 = df[df[cat_col] == groups[1]][num_col]

        t_stat, p_value = stats.ttest_ind(group1, group2)

        pooled_std = np.sqrt((group1.std() ** 2 + group2.std() ** 2) / 2)
        cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std != 0 else 0

        return {
            'statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'effect_size': cohens_d,
            'group1_mean': group1.mean(),
            'group2_mean': group2.mean(),
            'group1_std': group1.std(),
            'group2_std': group2.std(),
            'n1': len(group1),
            'n2': len(group2)
        }

    def _chi_square(self, hypothesis: Dict, dataset: Dataset) -> Dict[str, Any]:
        """Критерий хи-квадрат."""
        columns = self._get_columns(hypothesis, 2)
        col1, col2 = columns

        df = dataset.df.dropna(subset=[col1, col2])

        contingency = pd.crosstab(df[col1], df[col2])

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

        return {
            'statistic': chi2,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'effect_size': cramers_v,
            'degrees_of_freedom': dof,
            'contingency_table': contingency.to_dict(),
            'expected_frequencies': expected.tolist(),
            'n': n
        }

    def _shapiro_wilk(self, hypothesis: Dict, dataset: Dataset) -> Dict[str, Any]:
        """Тест Шапиро-Уилка на нормальность."""
        columns = self._get_columns(hypothesis, 1)
        col = columns[0]

        data = dataset.df[col].dropna()

        if len(data) > 5000:
            data = data.sample(5000, random_state=42)

        if len(data) < 3:
            raise ValueError(f"Недостаточно данных для теста нормальности: {len(data)} наблюдений")

        stat, p_value = stats.shapiro(data)

        return {
            'statistic': stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'effect_size': 1 - stat,
            'n': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        }

    def _bootstrap_ci(self, data1, data2, stat_func, n_bootstrap=1000):
        """Бутстрап доверительного интервала."""
        if len(data1) < 10 or len(data2) < 10:
            return None

        correlations = []
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(data1, size=len(data1), replace=True)
            sample2 = np.random.choice(data2, size=len(data2), replace=True)
            corr, _ = stat_func(sample1, sample2)
            correlations.append(corr)

        ci_lower = np.percentile(correlations, 2.5)
        ci_upper = np.percentile(correlations, 97.5)

        return [float(ci_lower), float(ci_upper)]

    def _get_columns(self, hypothesis: Dict, expected_count: int) -> list:
        """Получение колонок из гипотезы."""
        columns = hypothesis.get('columns', [])

        if len(columns) < expected_count:
            raise ValueError(f"Требуется {expected_count} колонок, найдено {len(columns)}")

        return columns[:expected_count]

    def _unknown_method(self, hypothesis: Dict, method: str) -> Dict[str, Any]:
        """Обработка неизвестного метода."""
        return {
            'statistic': None,
            'p_value': None,
            'is_significant': False,
            'error': f"Неизвестный метод: {method}"
        }

    def _error_result(self, hypothesis: Dict, method: str, error: str) -> Dict[str, Any]:
        """Результат с ошибкой."""
        return {
            'statistic': None,
            'p_value': None,
            'is_significant': False,
            'error': error
        }

    def _spearman_correlation(self, hypothesis: Dict, dataset: Dataset) -> Dict[str, Any]:
        """Корреляция Спирмена."""
        pass

    def _anova(self, hypothesis: Dict, dataset: Dataset) -> Dict[str, Any]:
        """Дисперсионный анализ."""
        pass

