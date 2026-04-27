"""Селектор методов анализа."""

from typing import Dict, Any, List
from ..config import AgentConfig
from ..data.loader import Dataset
from ..utils.logger import logger


class MethodSelector:
    """Селектор статистических методов."""
    
    def __init__(self, config: AgentConfig):
        self.config = config

    def select(self, hypothesis: Dict[str, Any], dataset: Dataset) -> str:
        """Выбор метода анализа."""
        hyp_type = hypothesis['type']
        columns = hypothesis['columns']

        for col in columns:
            if col not in dataset.df.columns:
                logger.warning(f"Колонка {col} не найдена")
                return 'unknown'

        # Маппинг типов на методы
        type_to_method = {
            'correlation': 'pearson',
            'mean_difference': self._select_mean_test(dataset, columns),
            'normality': 'shapiro',
            'independence': 'chi2',
            'regression': 'regression',
            'clustering': 'clustering',
            'segmentation': 'segmentation',  # НОВОЕ
            'interaction_effect': 'interaction',  # НОВОЕ
            'threshold_effect': 'threshold',  # НОВОЕ
            'nonlinear_relationship': 'nonlinear',  # НОВОЕ
            'mediation_analysis': 'mediation'  # НОВОЕ
        }

        return type_to_method.get(hyp_type, 'unknown')
    
    def _select_mean_test(self, dataset: Dataset, columns: List[str]) -> str:
        """Выбор метода для сравнения средних."""
        if len(columns) != 2:
            return 'unknown'
        
        cat_col = [c for c in columns if c in dataset.metadata['categorical_columns']]
        if not cat_col:
            return 'unknown'
        
        n_groups = dataset.df[cat_col[0]].nunique()
        
        if n_groups == 2:
            return 't_test'
        elif n_groups > 2:
            return 'anova'
        else:
            return 'unknown'
