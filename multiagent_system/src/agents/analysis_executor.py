"""Исполнитель анализа с расширенными методами."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, silhouette_score
from typing import Dict, Any
from ..config import AgentConfig
from ..data.loader import Dataset
from ..utils.logger import logger


class AnalysisExecutor:
    """Исполнитель статистического анализа."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
    
    def execute(self, hypothesis: Dict[str, Any], method: str, 
                dataset: Dataset) -> Dict[str, Any]:
        """Выполнение анализа."""
        try:
            if method == 'pearson':
                return self._pearson_correlation(hypothesis, dataset)
            elif method == 't_test':
                return self._t_test(hypothesis, dataset)
            elif method == 'anova':
                return self._anova(hypothesis, dataset)
            elif method == 'shapiro':
                return self._shapiro_test(hypothesis, dataset)
            elif method == 'chi2':
                return self._chi2_test(hypothesis, dataset)
            elif method == 'regression':
                return self._linear_regression(hypothesis, dataset)
            elif method == 'clustering':
                return self._kmeans_clustering(hypothesis, dataset)
            elif method == 'segmentation':
                return self._segmentation_analysis(hypothesis, dataset)
            elif method == 'interaction':
                return self._interaction_analysis(hypothesis, dataset)
            elif method == 'threshold':
                return self._threshold_analysis(hypothesis, dataset)
            elif method == 'nonlinear':
                return self._nonlinear_analysis(hypothesis, dataset)
            elif method == 'mediation':
                return self._mediation_analysis(hypothesis, dataset)
            else:
                return {'error': f'Неизвестный метод: {method}', 'is_significant': False}
        
        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            return {'error': str(e), 'is_significant': False}
    
    def _pearson_correlation(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """Корреляция Пирсона."""
        col1, col2 = hypothesis['columns']
        data = dataset.df[[col1, col2]].dropna()
        
        if len(data) < 3:
            return {'error': 'Недостаточно данных', 'is_significant': False}
        
        corr, p_value = stats.pearsonr(data[col1], data[col2])
        
        return {
            'method': 'pearson',
            'statistic': float(corr),
            'p_value': float(p_value),
            'is_significant': p_value < self.config.significance_level,
            'sample_size': len(data),
            'correlation_strength': self._interpret_correlation(abs(corr))
        }
    
    def _t_test(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """T-тест."""
        num_col = [c for c in hypothesis['columns'] 
                   if c in dataset.metadata['numeric_columns']][0]
        cat_col = [c for c in hypothesis['columns'] 
                   if c in dataset.metadata['categorical_columns']][0]
        
        groups = dataset.df[cat_col].unique()
        if len(groups) != 2:
            return {'error': 'Требуется 2 группы', 'is_significant': False}
        
        group1 = dataset.df[dataset.df[cat_col] == groups[0]][num_col].dropna()
        group2 = dataset.df[dataset.df[cat_col] == groups[1]][num_col].dropna()
        
        if len(group1) < 2 or len(group2) < 2:
            return {'error': 'Недостаточно данных', 'is_significant': False}
        
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Cohen's d
        pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
        cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
        
        return {
            'method': 't_test',
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': p_value < self.config.significance_level,
            'sample_size': len(group1) + len(group2),
            'effect_size': float(cohens_d),
            'group1_mean': float(group1.mean()),
            'group2_mean': float(group2.mean())
        }
    
    def _anova(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """ANOVA."""
        num_col = [c for c in hypothesis['columns'] 
                   if c in dataset.metadata['numeric_columns']][0]
        cat_col = [c for c in hypothesis['columns'] 
                   if c in dataset.metadata['categorical_columns']][0]
        
        groups = [dataset.df[dataset.df[cat_col] == g][num_col].dropna() 
                 for g in dataset.df[cat_col].unique()]
        groups = [g for g in groups if len(g) >= 2]
        
        if len(groups) < 2:
            return {'error': 'Недостаточно групп', 'is_significant': False}
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            'method': 'anova',
            'statistic': float(f_stat),
            'p_value': float(p_value),
            'is_significant': p_value < self.config.significance_level,
            'sample_size': sum(len(g) for g in groups),
            'n_groups': len(groups)
        }
    
    def _shapiro_test(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """Тест Шапиро-Уилка."""
        col = hypothesis['columns'][0]
        data = dataset.df[col].dropna()
        
        if len(data) < 3:
            return {'error': 'Недостаточно данных', 'is_significant': False}
        
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)
        
        stat, p_value = stats.shapiro(data)
        
        return {
            'method': 'shapiro',
            'statistic': float(stat),
            'p_value': float(p_value),
            'is_significant': p_value < self.config.significance_level,
            'sample_size': len(data)
        }
    
    def _chi2_test(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """Хи-квадрат тест."""
        col1, col2 = hypothesis['columns']
        contingency = pd.crosstab(dataset.df[col1], dataset.df[col2])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        return {
            'method': 'chi2',
            'statistic': float(chi2),
            'p_value': float(p_value),
            'is_significant': p_value < self.config.significance_level,
            'sample_size': int(contingency.sum().sum()),
            'degrees_of_freedom': int(dof)
        }
    
    def _linear_regression(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """Линейная регрессия."""
        columns = hypothesis['columns']
        target = columns[0]
        predictors = columns[1:]
        
        data = dataset.df[columns].dropna()
        
        if len(data) < 10:
            return {'error': 'Недостаточно данных', 'is_significant': False}
        
        X = data[predictors].values
        y = data[target].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # F-статистика для регрессии
        n = len(data)
        k = len(predictors)
        f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
        p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
        
        return {
            'method': 'regression',
            'statistic': float(r2),
            'p_value': float(p_value),
            'is_significant': p_value < self.config.significance_level,
            'sample_size': n,
            'r_squared': float(r2),
            'coefficients': [float(c) for c in model.coef_]
        }
    
    def _kmeans_clustering(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """K-means кластеризация."""
        columns = hypothesis['columns']
        data = dataset.df[columns].dropna()
        
        if len(data) < 10:
            return {'error': 'Недостаточно данных', 'is_significant': False}
        
        # Пробуем 2-4 кластера
        best_score = -1
        best_k = 2
        
        for k in range(2, min(5, len(data) // 3)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Итоговая кластеризация
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        return {
            'method': 'clustering',
            'statistic': float(best_score),
            'p_value': 0.0,  # Для кластеризации p-value не применим
            'is_significant': best_score > 0.3,  # Пороговое значение
            'sample_size': len(data),
            'n_clusters': best_k,
            'silhouette_score': float(best_score)
        }

    def _segmentation_analysis(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """Сегментация с профилированием."""
        from sklearn.preprocessing import StandardScaler

        columns = hypothesis['columns']
        data = dataset.df[columns].dropna()

        if len(data) < 20:
            return {'error': 'Недостаточно данных', 'is_significant': False}

        # Только числовые для кластеризации
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {'error': 'Нет числовых данных', 'is_significant': False}

        X_scaled = StandardScaler().fit_transform(numeric_data)

        # Поиск оптимального k
        best_k = 2
        best_score = -1

        for k in range(2, min(6, len(data) // 10)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        # Финальная кластеризация
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Профилирование сегментов
        data_with_labels = numeric_data.copy()
        data_with_labels['segment'] = labels

        profiles = []
        for seg in range(best_k):
            seg_data = data_with_labels[data_with_labels['segment'] == seg]
            profile = {
                'segment': int(seg),
                'size': int(len(seg_data)),
                'means': {col: float(seg_data[col].mean()) for col in numeric_data.columns}
            }
            profiles.append(profile)

        return {
            'method': 'segmentation',
            'statistic': float(best_score),
            'p_value': 0.0,  # N/A для кластеризации
            'is_significant': best_score > 0.3,
            'sample_size': len(data),
            'n_clusters': best_k,
            'silhouette_score': float(best_score),
            'segment_profiles': profiles
        }

    def _interaction_analysis(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """Анализ взаимодействий."""
        from sklearn.preprocessing import StandardScaler

        columns = hypothesis['columns']
        if len(columns) < 3:
            return {'error': 'Требуется минимум 3 переменные', 'is_significant': False}

        target = columns[0]
        predictors = columns[1:3]

        data = dataset.df[[target] + predictors].dropna()

        if len(data) < 30:
            return {'error': 'Недостаточно данных', 'is_significant': False}

        # Модель без взаимодействия
        X_base = data[predictors].values
        y = data[target].values

        model_base = LinearRegression()
        model_base.fit(X_base, y)
        r2_base = r2_score(y, model_base.predict(X_base))

        # Модель с взаимодействием
        interaction_term = data[predictors[0]] * data[predictors[1]]
        X_interaction = np.column_stack([X_base, interaction_term])

        model_int = LinearRegression()
        model_int.fit(X_interaction, y)
        r2_int = r2_score(y, model_int.predict(X_interaction))

        # F-test для сравнения моделей
        n = len(data)
        k1 = X_base.shape[1]
        k2 = X_interaction.shape[1]

        f_stat = ((r2_int - r2_base) / (k2 - k1)) / ((1 - r2_int) / (n - k2 - 1))
        p_value = 1 - stats.f.cdf(f_stat, k2 - k1, n - k2 - 1)

        return {
            'method': 'interaction',
            'statistic': float(r2_int - r2_base),
            'p_value': float(p_value),
            'is_significant': p_value < self.config.significance_level,
            'sample_size': n,
            'r2_base': float(r2_base),
            'r2_with_interaction': float(r2_int),
            'improvement': float(r2_int - r2_base)
        }

    def _threshold_analysis(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """Поиск пороговых эффектов (упрощенная версия)."""
        col = hypothesis['columns'][0]
        data = dataset.df[[col]].dropna()

        if len(data) < 50:
            return {'error': 'Недостаточно данных', 'is_significant': False}

        # Поиск скачков в данных через квантили
        quantiles = [0.25, 0.5, 0.75]
        thresholds = [data[col].quantile(q) for q in quantiles]

        # Простая эвристика: большой разброс = есть пороги
        variance = data[col].var()
        mean = data[col].mean()
        cv = variance / mean if mean != 0 else 0

        is_significant = cv > 0.5  # Коэффициент вариации

        return {
            'method': 'threshold',
            'statistic': float(cv),
            'p_value': 0.05 if is_significant else 0.5,
            'is_significant': is_significant,
            'sample_size': len(data),
            'suggested_thresholds': [float(t) for t in thresholds]
        }

    def _nonlinear_analysis(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """Анализ нелинейных зависимостей (полиномиальная регрессия)."""
        from sklearn.preprocessing import PolynomialFeatures

        col1, col2 = hypothesis['columns']
        data = dataset.df[[col1, col2]].dropna()

        if len(data) < 30:
            return {'error': 'Недостаточно данных', 'is_significant': False}

        X = data[[col1]].values
        y = data[col2].values

        # Линейная модель
        model_linear = LinearRegression()
        model_linear.fit(X, y)
        r2_linear = r2_score(y, model_linear.predict(X))

        # Полиномиальная модель (степень 2)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model_poly = LinearRegression()
        model_poly.fit(X_poly, y)
        r2_poly = r2_score(y, model_poly.predict(X_poly))

        improvement = r2_poly - r2_linear

        # F-test
        n = len(data)
        k1 = 1
        k2 = 2

        f_stat = ((r2_poly - r2_linear) / (k2 - k1)) / ((1 - r2_poly) / (n - k2 - 1))
        p_value = 1 - stats.f.cdf(f_stat, k2 - k1, n - k2 - 1)

        return {
            'method': 'nonlinear',
            'statistic': float(improvement),
            'p_value': float(p_value),
            'is_significant': p_value < self.config.significance_level and improvement > 0.05,
            'sample_size': n,
            'r2_linear': float(r2_linear),
            'r2_polynomial': float(r2_poly),
            'improvement': float(improvement)
        }

    def _mediation_analysis(self, hypothesis: Dict, dataset: Dataset) -> Dict:
        """Базовый медиационный анализ."""
        if len(hypothesis['columns']) < 3:
            return {'error': 'Требуется 3 переменные', 'is_significant': False}

        predictor, mediator, outcome = hypothesis['columns']
        data = dataset.df[[predictor, mediator, outcome]].dropna()

        if len(data) < 50:
            return {'error': 'Недостаточно данных', 'is_significant': False}

        # Шаг 1: X -> Y (total effect)
        X = data[[predictor]].values
        y_outcome = data[outcome].values

        model_total = LinearRegression()
        model_total.fit(X, y_outcome)
        total_effect = model_total.coef_[0]

        # Шаг 2: X -> M
        y_mediator = data[mediator].values
        model_a = LinearRegression()
        model_a.fit(X, y_mediator)
        a_path = model_a.coef_[0]

        # Шаг 3: X + M -> Y
        X_both = data[[predictor, mediator]].values
        model_b = LinearRegression()
        model_b.fit(X_both, y_outcome)
        b_path = model_b.coef_[1]  # Эффект медиатора
        direct_effect = model_b.coef_[0]  # Прямой эффект X

        # Indirect effect
        indirect_effect = a_path * b_path

        # Процент медиации
        pct_mediated = abs(indirect_effect / total_effect) if total_effect != 0 else 0

        # Простой тест значимости (упрощенный)
        is_significant = pct_mediated > 0.2  # Хотя бы 20% эффекта через медиатор

        return {
            'method': 'mediation',
            'statistic': float(pct_mediated),
            'p_value': 0.05 if is_significant else 0.5,
            'is_significant': is_significant,
            'sample_size': len(data),
            'total_effect': float(total_effect),
            'direct_effect': float(direct_effect),
            'indirect_effect': float(indirect_effect),
            'percent_mediated': float(pct_mediated * 100)
        }

    @staticmethod
    def _interpret_correlation(corr: float) -> str:
        """Интерпретация силы корреляции."""
        corr = abs(corr)
        if corr > 0.7:
            return "сильная"
        elif corr > 0.4:
            return "умеренная"
        elif corr > 0.2:
            return "слабая"
        else:
            return "очень слабая"
