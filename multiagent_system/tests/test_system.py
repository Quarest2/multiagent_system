"""
Тесты мультиагентной системы.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.config import SystemConfig, AgentConfig, DataConfig
from src.data.loader import DataLoader, Dataset
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.method_selector import MethodSelector
from src.agents.analysis_executor import AnalysisExecutor
from src.core.orchestrator import Orchestrator


@pytest.fixture
def sample_data():
    """Тестовые данные."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric1': np.random.normal(0, 1, 100),
        'numeric2': np.random.normal(5, 2, 100),
        'category1': np.random.choice(['A', 'B'], 100),
        'category2': np.random.choice(['X', 'Y', 'Z'], 100)
    })


@pytest.fixture
def config():
    """Тестовая конфигурация."""
    return SystemConfig.default()


def test_data_loader(sample_data, tmp_path):
    """Тест загрузчика данных."""
    # Сохраняем данные
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)

    # Загружаем
    loader = DataLoader()
    dataset = loader.load(str(data_path))

    assert isinstance(dataset, Dataset)
    assert len(dataset.df) == 100
    assert len(dataset.metadata['numeric_columns']) == 2
    assert len(dataset.metadata['categorical_columns']) == 2


def test_hypothesis_generator(sample_data, config):
    """Тест генератора гипотез."""
    dataset = Dataset(sample_data)
    generator = HypothesisGenerator(config.agent)

    hypotheses = generator.generate(dataset)

    assert len(hypotheses) > 0
    assert len(hypotheses) <= config.agent.max_hypotheses
    assert all('text' in h for h in hypotheses)
    assert all('type' in h for h in hypotheses)


def test_method_selector(sample_data, config):
    """Тест селектора методов."""
    dataset = Dataset(sample_data)
    selector = MethodSelector(config.agent)

    # Тест корреляции
    hyp_corr = {
        'type': 'correlation',
        'columns': ['numeric1', 'numeric2']
    }
    method = selector.select(hyp_corr, dataset)
    assert method == 'pearson'

    # Тест сравнения средних
    hyp_mean = {
        'type': 'mean_difference',
        'columns': ['numeric1', 'category1']
    }
    method = selector.select(hyp_mean, dataset)
    assert method == 't_test'


def test_analysis_executor(sample_data, config):
    """Тест исполнителя анализа."""
    dataset = Dataset(sample_data)
    executor = AnalysisExecutor(config.agent)

    # Тест корреляции
    hypothesis = {
        'type': 'correlation',
        'columns': ['numeric1', 'numeric2']
    }

    result = executor.execute(hypothesis, 'pearson', dataset)

    assert 'p_value' in result
    assert 'statistic' in result
    assert 'is_significant' in result
    assert isinstance(result['p_value'], float)


def test_orchestrator_run(sample_data, tmp_path, config):
    """Тест полного запуска системы."""
    # Сохраняем данные
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)

    # Запускаем систему
    orchestrator = Orchestrator(config)
    results = orchestrator.run(str(data_path))

    assert len(results) > 0
    assert all('hypothesis_text' in r for r in results)
    assert all('p_value' in r for r in results)
    assert all('is_significant' in r for r in results)


def test_save_results(sample_data, tmp_path, config):
    """Тест сохранения результатов."""
    # Сохраняем данные
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)

    # Запускаем систему
    orchestrator = Orchestrator(config)
    results = orchestrator.run(str(data_path))

    # Сохраняем результаты
    output_path = tmp_path / "results.json"
    orchestrator.save_results(str(output_path))

    assert output_path.exists()
    assert (tmp_path / "results.csv").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
