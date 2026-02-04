"""
Интеграционные тесты системы.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import patch, MagicMock

from src.config import ConfigManager
from src.core.orchestrator import Orchestrator
from src.data.loader import DataLoader
from src.agents.hypothesis_generator import HypothesisGenerator
from src.utils.metrics import MetricsCollector


class TestSystemIntegration:
    """Интеграционные тесты системы."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = ConfigManager()

        # Создаем реалистичный тестовый датасет
        np.random.seed(42)
        n_samples = 100

        self.test_data = pd.DataFrame({
            'age': np.random.normal(35, 10, n_samples).clip(18, 70),
            'income': np.random.lognormal(10, 0.5, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'satisfaction': np.random.randint(1, 6, n_samples),
            'tenure': np.random.exponential(5, n_samples).clip(0, 30),
            'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], n_samples),
            'performance': np.random.normal(75, 15, n_samples).clip(0, 100)
        })

    def test_end_to_end_without_llm(self):
        """End-to-end тест без LLM."""
        # Отключаем LLM для скорости
        self.config.update_llm_config(enable=False)
        self.config.update_agent_config(max_hypotheses=5, refinement_cycles=1)

        # Сохраняем тестовые данные во временный файл
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            filepath = f.name

        try:
            # Создаем и запускаем оркестратор
            orchestrator = Orchestrator(self.config, verbose=False)
            results = orchestrator.run(filepath)

            # Проверяем результаты
            assert isinstance(results, list)
            assert len(results) > 0

            # Проверяем структуру результатов
            for result in results:
                assert 'hypothesis_text' in result
                assert 'method' in result
                assert 'p_value' in result or result['p_value'] is None
                assert 'is_significant' in result
                assert 'conclusion' in result
                assert 'quality_score' in result

                # Проверяем, что p-value в допустимом диапазоне
                if result['p_value'] is not None:
                    assert 0 <= result['p_value'] <= 1

            # Проверяем метрики
            metrics = orchestrator.get_status()['metrics']
            assert metrics['total_hypotheses'] == len(results)

        finally:
            Path(filepath).unlink()

    def test_end_to_end_with_llm_mock(self):
        """End-to-end тест с моком LLM."""
        # Включаем LLM, но мокаем вызовы
        self.config.update_llm_config(enable=True)
        self.config.update_agent_config(max_hypotheses=3, refinement_cycles=2)

        # Мокаем все LLM вызовы
        with patch('src.llm.openai_client.OpenAIClient.generate') as mock_llm:
            # Настраиваем мок для разных типов запросов
            def llm_side_effect(prompt, **kwargs):
                if 'гипотез' in prompt:
                    # Для генерации гипотез
                    return {
                        'hypotheses': [
                            {
                                'id': 1,
                                'text': 'Доход коррелирует с уровнем образования',
                                'type': 'correlation',
                                'columns': ['income', 'education'],
                                'reasoning': 'Обычно с образованием растет доход',
                                'expected_method': 'spearman'
                            },
                            {
                                'id': 2,
                                'text': 'Средняя удовлетворенность отличается по отделам',
                                'type': 'mean_difference',
                                'columns': ['satisfaction', 'department'],
                                'reasoning': 'Культура компании различается по отделам',
                                'expected_method': 'anova'
                            }
                        ]
                    }
                elif 'интерпретац' in prompt.lower():
                    # Для интерпретации
                    return {
                        'conclusion': 'Гипотеза подтверждена',
                        'explanation': 'Результат статистически значим',
                        'confidence': 0.85
                    }
                elif 'качеств' in prompt.lower():
                    # Для проверки качества
                    return {
                        'score': 0.8,
                        'issues': [],
                        'suggestions': [],
                        'needs_refinement': False
                    }
                elif 'дорабо' in prompt.lower():
                    # Для доработки
                    return {
                        'hypotheses': [{
                            'text': 'Уточненная гипотеза',
                            'type': 'correlation',
                            'columns': ['income', 'education'],
                            'improvements': ['Уточнена формулировка']
                        }]
                    }
                else:
                    # Для выбора метода
                    return {
                        'methods': [
                            {'name': 'spearman', 'reasoning': 'Порядковые данные'}
                        ]
                    }

            mock_llm.side_effect = llm_side_effect

            # Сохраняем тестовые данные
            with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
                self.test_data.to_csv(f.name, index=False)
                filepath = f.name

            try:
                orchestrator = Orchestrator(self.config, verbose=False)
                results = orchestrator.run(filepath)

                # Проверяем результаты
                assert len(results) > 0

                # Проверяем, что LLM вызывался
                assert mock_llm.call_count > 0

                # Проверяем, что есть LLM-enhanced гипотезы
                llm_enhanced = [r for r in results if r.get('llm_enhanced', False)]
                assert len(llm_enhanced) > 0

            finally:
                Path(filepath).unlink()

    def test_system_with_different_datasets(self):
        """Тест системы с разными датасетами."""
        datasets = [
            # 1. Простой датасет с четкой корреляцией
            pd.DataFrame({
                'x': np.arange(100),
                'y': np.arange(100) * 2 + np.random.normal(0, 5, 100),
                'group': ['A'] * 50 + ['B'] * 50
            }),

            # 2. Датисет с категориальными переменными
            pd.DataFrame({
                'category': np.random.choice(['A', 'B', 'C'], 100),
                'value': np.random.normal(0, 1, 100),
                'score': np.random.randint(1, 11, 100)
            }),

            # 3. Датисет с временным рядом
            pd.DataFrame({
                'time': pd.date_range('2023-01-01', periods=100, freq='D'),
                'value': np.cumsum(np.random.normal(0, 1, 100)),
                'event': [0] * 50 + [1] * 50
            })
        ]

        self.config.update_llm_config(enable=False)
        self.config.update_agent_config(max_hypotheses=3)

        for i, df in enumerate(datasets):
            with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
                df.to_csv(f.name, index=False)
                filepath = f.name

            try:
                orchestrator = Orchestrator(self.config, verbose=False)
                results = orchestrator.run(filepath)

                # Проверяем, что система работает с разными датасетами
                assert len(results) > 0

                # Сохраняем результаты для анализа
                output_file = f'test_dataset_{i}_results.json'
                orchestrator.save_results(results, output_file)

                # Проверяем, что файл создан
                assert Path(output_file).exists()
                Path(output_file).unlink()  # Очищаем

            finally:
                Path(filepath).unlink()

    def test_error_handling(self):
        """Тест обработки ошибок в системе."""
        # Создаем проблемный датасет
        problematic_data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': ['A', 'B', 'C', None, 'E'],
            'col3': [0, 0, 0, 0, 0]  # Константная колонка
        })

        self.config.update_llm_config(enable=False)

        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            problematic_data.to_csv(f.name, index=False)
            filepath = f.name

        try:
            orchestrator = Orchestrator(self.config, verbose=False)
            results = orchestrator.run(filepath)

            # Система должна обработать проблемные данные
            assert isinstance(results, list)

            # Может быть пусто, если нет валидных гипотез
            # Главное - не должно быть исключений

        except Exception as e:
            # Допускаются только определенные ошибки
            pytest.fail(f"Система не должна падать на проблемных данных: {e}")

        finally:
            Path(filepath).unlink()

    def test_metrics_collection(self):
        """Тест сбора метрик."""
        self.config.update_llm_config(enable=False)

        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            filepath = f.name

        try:
            orchestrator = Orchestrator(self.config, verbose=False)
            results = orchestrator.run(filepath)

            # Проверяем метрики
            metrics = orchestrator.get_status()['metrics']

            assert 'total_hypotheses' in metrics
            assert metrics['total_hypotheses'] == len(results)

            if len(results) > 0:
                assert 'significant_hypotheses' in metrics
                assert 'avg_quality_score' in metrics
                assert 0 <= metrics['avg_quality_score'] <= 1

            # Проверяем workflow статус
            workflow_status = orchestrator.workflow.get_status()
            assert workflow_status['status'] == 'completed'

        finally:
            Path(filepath).unlink()

    def test_configuration_variations(self):
        """Тест различных конфигураций системы."""
        configs = [
            # Базовая конфигурация
            {'max_hypotheses': 5, 'refinement_cycles': 1, 'enable_llm': False},

            # С LLM
            {'max_hypotheses': 3, 'refinement_cycles': 2, 'enable_llm': True},

            # Агрессивная генерация
            {'max_hypotheses': 10, 'refinement_cycles': 0, 'enable_llm': False},

            # Тщательная доработка
            {'max_hypotheses': 2, 'refinement_cycles': 3, 'enable_llm': False}
        ]

        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            filepath = f.name

        try:
            results_by_config = []

            for config in configs:
                # Создаем новую конфигурацию
                test_config = ConfigManager()
                test_config.update_agent_config(
                    max_hypotheses=config['max_hypotheses'],
                    refinement_cycles=config['refinement_cycles']
                )
                test_config.update_llm_config(enable=config['enable_llm'])

                # Мокаем LLM если включен
                if config['enable_llm']:
                    with patch('src.llm.openai_client.OpenAIClient.generate') as mock_llm:
                        mock_llm.return_value = {'hypotheses': []}
                        orchestrator = Orchestrator(test_config, verbose=False)
                        results = orchestrator.run(filepath)
                else:
                    orchestrator = Orchestrator(test_config, verbose=False)
                    results = orchestrator.run(filepath)

                results_by_config.append({
                    'config': config,
                    'results': results,
                    'metrics': orchestrator.get_status()['metrics']
                })

            # Проверяем, что все конфигурации работают
            assert len(results_by_config) == len(configs)

            # Анализируем различия
            for result in results_by_config:
                assert result['metrics']['total_hypotheses'] <= result['config']['max_hypotheses']

        finally:
            Path(filepath).unlink()

    def test_save_and_load_results(self):
        """Тест сохранения и загрузки результатов."""
        self.config.update_llm_config(enable=False)

        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            filepath = f.name

        try:
            orchestrator = Orchestrator(self.config, verbose=False)
            results = orchestrator.run(filepath)

            # Сохраняем результаты
            with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
                save_path = f.name

            orchestrator.save_results(results, save_path)

            # Проверяем, что файл создан и содержит данные
            assert Path(save_path).exists()

            with open(save_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)

            assert 'results' in saved_data
            assert 'metrics' in saved_data
            assert 'workflow_status' in saved_data

            # Проверяем, что все результаты сохранены
            assert len(saved_data['results']) == len(results)

            Path(save_path).unlink()

        finally:
            Path(filepath).unlink()

    def test_performance_benchmark(self):
        """Тест производительности системы."""
        import time

        # Создаем большой датасет для теста производительности
        large_data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, 1000)
            for i in range(20)
        })

        # Добавляем несколько категориальных колонок
        large_data['category'] = np.random.choice(['A', 'B', 'C', 'D'], 1000)
        large_data['target'] = np.random.normal(0, 1, 1000)

        self.config.update_llm_config(enable=False)
        self.config.update_agent_config(max_hypotheses=10)

        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            large_data.to_csv(f.name, index=False)
            filepath = f.name

        try:
            # Замеряем время выполнения
            start_time = time.time()

            orchestrator = Orchestrator(self.config, verbose=False)
            results = orchestrator.run(filepath)

            end_time = time.time()
            execution_time = end_time - start_time

            # Проверяем производительность
            assert len(results) > 0

            # Время на гипотезу
            time_per_hypothesis = execution_time / len(results)

            # Логируем метрики производительности
            print(f"\nПроизводительность системы:")
            print(f"  Всего гипотез: {len(results)}")
            print(f"  Общее время: {execution_time:.2f} сек")
            print(f"  Время на гипотезу: {time_per_hypothesis:.3f} сек")
            print(f"  Гипотез в секунду: {len(results) / execution_time:.2f}")

            # Проверяем разумные лимиты производительности
            # (это субъективные тесты, можно настроить под ваши требования)
            assert execution_time < 30  # Должно выполняться менее 30 секунд
            assert time_per_hypothesis < 2  # Менее 2 секунд на гипотезу

        finally:
            Path(filepath).unlink()


class TestDataPipeline:
    """Тесты пайплайна данных."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = ConfigManager()
        self.loader = DataLoader(self.config)

    def test_data_loading_pipeline(self):
        """Тест полного пайплайна загрузки данных."""
        # Создаем сложный тестовый датасет
        complex_data = pd.DataFrame({
            'int_col': list(range(100)),
            'float_col': np.random.uniform(0, 1, 100),
            'cat_col': np.random.choice(['A', 'B', 'C'], 100),
            'text_col': ['text_' + str(i) for i in range(100)],
            'missing_col': [1 if i % 10 == 0 else np.nan for i in range(100)],
            'constant_col': [5] * 100,
            'date_col': pd.date_range('2023-01-01', periods=100, freq='D')
        })

        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            complex_data.to_csv(f.name, index=False)
            filepath = f.name

        try:
            dataset = self.loader.load(filepath)

            # Проверяем, что данные загружены
            assert isinstance(dataset, type(self.loader).__module__.Dataset)
            assert len(dataset.df) == 100

            # Проверяем предобработку
            # Константная колонка должна быть удалена
            assert 'constant_col' not in dataset.df.columns

            # Пропуски должны быть обработаны
            assert dataset.df['missing_col'].isnull().sum() == 0

            # Проверяем метаданные
            assert 'metadata' in dataset.__dict__
            assert 'numeric_columns' in dataset.metadata
            assert 'categorical_columns' in dataset.metadata

        finally:
            Path(filepath).unlink()

    def test_large_file_handling(self):
        """Тест обработки больших файлов."""
        # Создаем большой файл (10К строк, 50 колонок)
        n_rows = 10000
        n_cols = 50

        large_data = pd.DataFrame(
            np.random.randn(n_rows, n_cols),
            columns=[f'col_{i}' for i in range(n_cols)]
        )

        large_data['category'] = np.random.choice(['A', 'B'], n_rows)

        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            # Сохраняем без индекса для экономии места
            large_data.to_csv(f.name, index=False)
            filepath = f.name

        try:
            import time
            start_time = time.time()

            dataset = self.loader.load(filepath)

            end_time = time.time()
            load_time = end_time - start_time

            # Проверяем, что данные загружены
            assert dataset.df.shape == (n_rows, n_cols + 1)

            # Логируем время загрузки
            print(f"\nЗагрузка большого файла:")
            print(f"  Размер: {n_rows} строк × {n_cols + 1} колонок")
            print(f"  Время загрузки: {load_time:.2f} сек")
            print(f"  Строк в секунду: {n_rows / load_time:.0f}")

            # Проверяем разумное время загрузки
            assert load_time < 10  # Должно грузиться менее 10 секунд

        finally:
            Path(filepath).unlink()


class TestAgentInteraction:
    """Тесты взаимодействия агентов."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = ConfigManager()
        self.config.update_llm_config(enable=False)

        # Создаем тестовые компоненты
        from src.agents.hypothesis_generator import HypothesisGenerator
        from src.agents.method_selector import MethodSelector
        from src.agents.analysis_executor import AnalysisExecutor
        from src.agents.interpreter import Interpreter

        self.generator = HypothesisGenerator(self.config)
        self.selector = MethodSelector(self.config)
        self.executor = AnalysisExecutor(self.config)
        self.interpreter = Interpreter(self.config)

        # Тестовый датасет
        self.test_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'target': np.random.choice([0, 1], 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })

        self.dataset = type(self.generator).__module__.Dataset(self.test_df)

    def test_agent_chain(self):
        """Тест цепочки агентов."""
        # Генерация гипотез
        hypotheses = self.generator.generate(self.dataset)
        assert len(hypotheses) > 0

        # Обрабатываем первую гипотезу
        hypothesis = hypotheses[0]

        # Выбор метода
        method = self.selector.select(hypothesis, self.dataset)
        assert method != 'unknown'

        # Выполнение анализа
        analysis_result = self.executor.execute(hypothesis, method, self.dataset)
        assert 'p_value' in analysis_result or 'error' in analysis_result

        # Интерпретация результатов
        if 'error' not in analysis_result:
            interpretation = self.interpreter.interpret(hypothesis, analysis_result)
            assert 'conclusion' in interpretation
            assert 'confidence' in interpretation

    def test_error_propagation(self):
        """Тест распространения ошибок между агентами."""
        # Создаем проблемную гипотезу
        hypothesis = {
            'text': 'Некорректная гипотеза',
            'type': 'invalid_type',
            'columns': ['non_existent_column']
        }

        # Выбор метода должен вернуть 'unknown'
        method = self.selector.select(hypothesis, self.dataset)
        assert method == 'unknown'

        # Исполнитель должен вернуть ошибку
        result = self.executor.execute(hypothesis, method, self.dataset)
        assert 'error' in result

        # Интерпретатор должен обработать ошибку
        interpretation = self.interpreter.interpret(hypothesis, result)
        assert 'Ошибка' in interpretation['conclusion']
        assert interpretation['confidence'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
