"""
Тесты загрузчика данных.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from src.data.loader import DataLoader, Dataset
from src.config import ConfigManager


class TestDataLoader:
    """Тесты загрузчика данных."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = ConfigManager()
        self.loader = DataLoader(self.config)

        # Создаем тестовые данные
        self.test_data = pd.DataFrame({
            'numeric_1': [1, 2, 3, 4, 5],
            'numeric_2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical_1': ['A', 'B', 'A', 'B', 'C'],
            'categorical_2': ['X', 'X', 'Y', 'Y', 'Z'],
            'mixed': [1, 'text', 3, 'text', 5]
        })

    def test_create_dataset(self):
        """Тест создания Dataset."""
        dataset = Dataset(self.test_data)

        assert dataset.df.shape == (5, 5)
        assert 'metadata' in dataset.__dict__
        assert 'shape' in dataset.metadata
        assert dataset.metadata['shape'] == (5, 5)

        # Проверяем определение типов колонок
        assert 'numeric_1' in dataset.metadata['numeric_columns']
        assert 'categorical_1' in dataset.metadata['categorical_columns']

    def test_metadata_extraction(self):
        """Тест извлечения метаданных."""
        dataset = Dataset(self.test_data)
        metadata = dataset.metadata

        # Проверяем основные поля
        assert metadata['shape'] == (5, 5)
        assert len(metadata['columns']) == 5
        assert 'numeric_1' in metadata['dtypes']
        assert 'basic_stats' in metadata

        # Проверяем статистику
        assert 'numeric' in metadata['basic_stats']
        assert 'categorical' in metadata['basic_stats']

    def test_load_csv(self):
        """Тест загрузки CSV."""
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            filepath = f.name

        try:
            dataset = self.loader.load(filepath)

            assert isinstance(dataset, Dataset)
            assert dataset.df.shape == (5, 5)
            assert list(dataset.df.columns) == list(self.test_data.columns)

        finally:
            Path(filepath).unlink()

    def test_load_excel(self):
        """Тест загрузки Excel."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            self.test_data.to_excel(f.name, index=False)
            filepath = f.name

        try:
            dataset = self.loader.load(filepath)

            assert isinstance(dataset, Dataset)
            assert dataset.df.shape == (5, 5)

        finally:
            Path(filepath).unlink()

    def test_load_json(self):
        """Тест загрузки JSON."""
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            self.test_data.to_json(f.name, orient='records')
            filepath = f.name

        try:
            dataset = self.loader.load(filepath)

            assert isinstance(dataset, Dataset)
            assert dataset.df.shape == (5, 5)

        finally:
            Path(filepath).unlink()

    def test_file_not_found(self):
        """Тест обработки отсутствующего файла."""
        with pytest.raises(FileNotFoundError):
            self.loader.load("non_existent_file.csv")

    def test_unsupported_format(self):
        """Тест неподдерживаемого формата."""
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as f:
            f.write("test data")
            filepath = f.name

        try:
            with pytest.raises(ValueError):
                self.loader.load(filepath)
        finally:
            Path(filepath).unlink()

    def test_preprocessing(self):
        """Тест предобработки данных в загрузчике."""
        # Создаем данные с пропусками и константными колонками
        test_data = pd.DataFrame({
            'numeric_1': [1, 2, np.nan, 4, 5],
            'numeric_2': [1.1, 1.1, 1.1, 1.1, 1.1],  # Константная
            'categorical_1': ['A', 'B', None, 'B', 'C'],
            'constant': [1, 1, 1, 1, 1]  # Константная
        })

        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            filepath = f.name

        try:
            dataset = self.loader.load(filepath)

            # Проверяем, что константные колонки удалены
            assert 'numeric_2' not in dataset.df.columns
            assert 'constant' not in dataset.df.columns

            # Проверяем, что пропуски обработаны
            assert dataset.df['numeric_1'].isnull().sum() == 0
            assert dataset.df['categorical_1'].isnull().sum() == 0

        finally:
            Path(filepath).unlink()

    def test_optimize_dtypes(self):
        """Тест оптимизации типов данных."""
        test_data = pd.DataFrame({
            'int8_range': pd.Series([1, 2, 3], dtype='int64'),
            'int16_range': pd.Series([1000, 2000, 3000], dtype='int64'),
            'uint8_range': pd.Series([1, 2, 3], dtype='int64'),
            'float_data': pd.Series([1.1, 2.2, 3.3], dtype='float64'),
            'object_data': pd.Series(['A', 'B', 'A'], dtype='object')
        })

        optimized = self.loader._optimize_dtypes(test_data)

        # Проверяем оптимизацию
        assert str(optimized['int8_range'].dtype) == 'int8'
        assert str(optimized['int16_range'].dtype) == 'int16'
        assert str(optimized['uint8_range'].dtype) == 'uint8'
        assert str(optimized['object_data'].dtype) == 'category'

    def test_dataset_methods(self):
        """Тест методов Dataset."""
        dataset = Dataset(self.test_data)

        # Проверяем get_numeric_columns
        numeric_cols = dataset._get_numeric_columns()
        assert 'numeric_1' in numeric_cols
        assert 'numeric_2' in numeric_cols
        assert 'categorical_1' not in numeric_cols

        # Проверяем get_categorical_columns
        categorical_cols = dataset._get_categorical_columns()
        assert 'categorical_1' in categorical_cols
        assert 'categorical_2' in categorical_cols
        assert 'numeric_1' not in categorical_cols

    def test_empty_dataset(self):
        """Тест пустого датасета."""
        empty_df = pd.DataFrame()
        dataset = Dataset(empty_df)

        assert dataset.df.shape == (0, 0)
        assert dataset.metadata['shape'] == (0, 0)
        assert dataset.metadata['numeric_columns'] == []
        assert dataset.metadata['categorical_columns'] == []

    def test_large_dataset(self):
        """Тест большого датасета."""
        # Создаем большой датасет
        large_data = pd.DataFrame({
            'col1': range(10000),
            'col2': [float(i) for i in range(10000)],
            'col3': ['category'] * 10000
        })

        dataset = Dataset(large_data)

        assert dataset.df.shape == (10000, 3)
        assert len(dataset.metadata['numeric_columns']) == 2
        assert len(dataset.metadata['categorical_columns']) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])