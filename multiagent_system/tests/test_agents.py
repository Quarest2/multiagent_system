"""
Тесты агентов системы.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.config import ConfigManager
from src.data.loader import Dataset
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.method_selector import MethodSelector
from src.agents.analysis_executor import AnalysisExecutor
from src.agents.interpreter import Interpreter
from src.agents.qa_inspector import QAInspector
from src.agents.refiner import Refiner


class TestHypothesisGenerator:
    """Тесты генератора гипотез."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = ConfigManager()
        self.generator = HypothesisGenerator(self.config)

        # Создаем тестовый датасет
        self.test_df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'salary': [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000],
            'department': ['IT', 'HR', 'IT', 'HR', 'IT', 'HR', 'IT', 'HR', 'IT', 'HR'],
            'experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'satisfaction': ['high', 'low', 'high', 'low', 'high', 'low', 'high', 'low', 'high', 'low']
        })

        self.dataset = Dataset(self.test_df)

    def test_generate_basic(self):
        """Тест базовой генерации гипотез."""
        hypotheses = self.generator.generate(self.dataset)

        assert isinstance(hypotheses, list)
        assert len(hypotheses) > 0

        # Проверяем структуру гипотез
        for hypothesis in hypotheses:
            assert 'id' in hypothesis
            assert 'text' in hypothesis
            assert 'type' in hypothesis
            assert 'columns' in hypothesis
            assert 'priority' in hypothesis

            # Проверяем, что текст не пустой
            assert isinstance(hypothesis['text'], str)
            assert len(hypothesis['text']) > 0

            # Проверяем, что тип гипотезы допустимый
            assert hypothesis['type'] in ['mean_difference', 'correlation',
                                          'distribution', 'trend', 'association']

    def test_generate_with_llm(self):
        """Тест генерации с использованием LLM."""
        # Мокаем LLM клиент
        with patch.object(self.generator.llm_client, 'generate') as mock_llm:
            mock_llm.return_value = {
                'hypotheses': [
                    {
                        'id': 1,
                        'text': 'Существует корреляция между возрастом и зарплатой',
                        'type': 'correlation',
                        'columns': ['age', 'salary'],
                        'reasoning': 'Обычно с возрастом растет зарплата',
                        'expected_method': 'pearson'
                    }
                ]
            }

            # Включаем LLM
            self.generator.config.llm.enable = True
            hypotheses = self.generator.generate(self.dataset)

            assert len(hypotheses) > 0
            mock_llm.assert_called_once()

    def test_generate_without_llm(self):
        """Тест генерации без LLM."""
        # Отключаем LLM
        self.generator.config.llm.enable = False
        hypotheses = self.generator.generate(self.dataset)

        assert len(hypotheses) > 0

        # Проверяем, что гипотезы имеют правильные колонки
        for hypothesis in hypotheses:
            columns = hypothesis.get('columns', [])
            for col in columns:
                assert col in self.test_df.columns

    def test_generate_max_hypotheses(self):
        """Тест ограничения количества гипотез."""
        # Устанавливаем ограничение
        self.generator.config.agents.max_hypotheses = 3
        hypotheses = self.generator.generate(self.dataset)

        assert len(hypotheses) <= 3

    def test_generate_empty_dataset(self):
        """Тест генерации для пустого датасета."""
        empty_df = pd.DataFrame()
        empty_dataset = Dataset(empty_df)

        hypotheses = self.generator.generate(empty_dataset)

        assert isinstance(hypotheses, list)
        assert len(hypotheses) == 0

    def test_generate_single_column(self):
        """Тест генерации для датасета с одной колонкой."""
        single_df = pd.DataFrame({'age': [25, 30, 35]})
        single_dataset = Dataset(single_df)

        hypotheses = self.generator.generate(single_dataset)

        # Должны быть только гипотезы о распределении
        for hypothesis in hypotheses:
            assert hypothesis['type'] == 'distribution'
            assert hypothesis['columns'] == ['age']

    def test_filter_hypotheses(self):
        """Тест фильтрации гипотез."""
        # Создаем тестовые гипотезы
        test_hypotheses = [
            {'id': 1, 'text': 'Test 1', 'type': 'correlation', 'columns': ['a', 'b'], 'priority': 1},
            {'id': 2, 'text': 'Test 2', 'type': 'mean_difference', 'columns': ['a', 'c'], 'priority': 2},
            {'id': 3, 'text': 'Test 3', 'type': 'distribution', 'columns': ['b'], 'priority': 3},
            {'id': 4, 'text': 'Test 4', 'type': 'trend', 'columns': ['c'], 'priority': 1},
            {'id': 5, 'text': 'Test 5', 'type': 'association', 'columns': ['a', 'd'], 'priority': 2}
        ]

        # Фильтруем до 3 гипотез
        filtered = self.generator.filter_hypotheses(test_hypotheses, max_hypotheses=3)

        assert len(filtered) == 3
        # Проверяем, что остались гипотезы с наивысшим приоритетом
        priorities = [h['priority'] for h in filtered]
        assert min(priorities) <= max(priorities)

    def test_generate_specific_types(self):
        """Тест генерации конкретных типов гипотез."""
        # Мокаем LLM для получения конкретных типов
        with patch.object(self.generator.llm_client, 'generate') as mock_llm:
            mock_llm.return_value = {
                'hypotheses': [
                    {
                        'id': 1,
                        'text': 'Средняя зарплата отличается по отделам',
                        'type': 'mean_difference',
                        'columns': ['salary', 'department'],
                        'reasoning': 'IT обычно платят больше',
                        'expected_method': 't_test'
                    },
                    {
                        'id': 2,
                        'text': 'Возраст коррелирует с опытом',
                        'type': 'correlation',
                        'columns': ['age', 'experience'],
                        'reasoning': 'С возрастом растет опыт',
                        'expected_method': 'pearson'
                    }
                ]
            }

            self.generator.config.llm.enable = True
            hypotheses = self.generator.generate(self.dataset)

            # Проверяем, что есть гипотезы обоих типов
            types = [h['type'] for h in hypotheses]
            assert 'mean_difference' in types
            assert 'correlation' in types


class TestMethodSelector:
    """Тесты селектора методов."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = ConfigManager()
        self.selector = MethodSelector(self.config)

        # Тестовый датасет
        self.test_df = pd.DataFrame({
            'numeric_1': np.random.normal(0, 1, 100),
            'numeric_2': np.random.normal(5, 2, 100),
            'categorical_2groups': np.random.choice(['A', 'B'], 100),
            'categorical_3groups': np.random.choice(['X', 'Y', 'Z'], 100),
            'binary': np.random.choice([0, 1], 100)
        })

        self.dataset = Dataset(self.test_df)

    def test_select_correlation(self):
        """Тест выбора метода для корреляции."""
        hypothesis = {
            'type': 'correlation',
            'columns': ['numeric_1', 'numeric_2']
        }

        method = self.selector.select(hypothesis, self.dataset)

        assert method in ['pearson', 'spearman', 'kendall']

    def test_select_mean_difference_2groups(self):
        """Тест выбора метода для сравнения средних (2 группы)."""
        hypothesis = {
            'type': 'mean_difference',
            'columns': ['numeric_1', 'categorical_2groups']
        }

        method = self.selector.select(hypothesis, self.dataset)

        assert method in ['t_test_ind', 'mann_whitney', 'welch']

    def test_select_mean_difference_3groups(self):
        """Тест выбора метода для сравнения средних (3+ группы)."""
        hypothesis = {
            'type': 'mean_difference',
            'columns': ['numeric_1', 'categorical_3groups']
        }

        method = self.selector.select(hypothesis, self.dataset)

        assert method in ['anova', 'kruskal_wallis']

    def test_select_distribution(self):
        """Тест выбора метода для проверки распределения."""
        hypothesis = {
            'type': 'distribution',
            'columns': ['numeric_1']
        }

        method = self.selector.select(hypothesis, self.dataset)

        assert method in ['shapiro', 'normaltest', 'anderson']

    def test_select_association(self):
        """Тест выбора метода для проверки ассоциации."""
        hypothesis = {
            'type': 'association',
            'columns': ['categorical_2groups', 'categorical_3groups']
        }

        method = self.selector.select(hypothesis, self.dataset)

        assert method in ['chi2', 'fisher_exact']

    def test_select_unknown(self):
        """Тест выбора метода для неизвестного типа гипотезы."""
        hypothesis = {
            'type': 'unknown_type',
            'columns': ['numeric_1', 'numeric_2']
        }

        method = self.selector.select(hypothesis, self.dataset)

        assert method == 'unknown'

    def test_select_with_llm(self):
        """Тест выбора метода с LLM."""
        with patch.object(self.selector.llm_client, 'generate') as mock_llm:
            mock_llm.return_value = {
                'methods': [
                    {'name': 'pearson', 'reasoning': 'Нормальное распределение'},
                    {'name': 'spearman', 'reasoning': 'Непараметрический'},
                    {'name': 'kendall', 'reasoning': 'Для малых выборок'}
                ]
            }

            self.selector.config.llm.enable = True
            hypothesis = {'type': 'correlation', 'columns': ['numeric_1', 'numeric_2']}
            method = self.selector.select(hypothesis, self.dataset)

            assert method in ['pearson', 'spearman', 'kendall']
            mock_llm.assert_called_once()

    def test_select_invalid_columns(self):
        """Тест выбора метода для несуществующих колонок."""
        hypothesis = {
            'type': 'correlation',
            'columns': ['non_existent_1', 'non_existent_2']
        }

        method = self.selector.select(hypothesis, self.dataset)

        assert method == 'unknown'


class TestAnalysisExecutor:
    """Тесты исполнителя анализа."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = ConfigManager()
        self.executor = AnalysisExecutor(self.config)

        # Тестовые данные
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'group_A': np.random.normal(10, 2, 50),
            'group_B': np.random.normal(12, 2, 50),
            'category': ['A'] * 50 + ['B'] * 50,
            'x': np.random.normal(0, 1, 100),
            'y': 2 * np.random.normal(0, 1, 100) + 0.5,
            'cat1': np.random.choice(['X', 'Y'], 100),
            'cat2': np.random.choice(['P', 'Q'], 100)
        })

        self.dataset = Dataset(self.test_df)

    def test_execute_t_test(self):
        """Тест выполнения t-теста."""
        hypothesis = {
            'type': 'mean_difference',
            'columns': ['group_A', 'category']
        }

        result = self.executor.execute(hypothesis, 't_test_ind', self.dataset)

        assert 'p_value' in result
        assert 'statistic' in result
        assert 'is_significant' in result
        assert isinstance(result['p_value'], (int, float))
        assert isinstance(result['statistic'], (int, float))
        assert isinstance(result['is_significant'], bool)

    def test_execute_pearson(self):
        """Тест выполнения корреляции Пирсона."""
        hypothesis = {
            'type': 'correlation',
            'columns': ['x', 'y']
        }

        result = self.executor.execute(hypothesis, 'pearson', self.dataset)

        assert 'p_value' in result
        assert 'correlation' in result
        assert isinstance(result['p_value'], (int, float))
        assert isinstance(result['correlation'], (int, float))

    def test_execute_chi2(self):
        """Тест выполнения критерия хи-квадрат."""
        hypothesis = {
            'type': 'association',
            'columns': ['cat1', 'cat2']
        }

        result = self.executor.execute(hypothesis, 'chi2', self.dataset)

        assert 'p_value' in result
        assert 'chi2_statistic' in result
        assert 'degrees_of_freedom' in result

    def test_execute_shapiro(self):
        """Тест выполнения теста Шапиро-Уилка."""
        hypothesis = {
            'type': 'distribution',
            'columns': ['x']
        }

        result = self.executor.execute(hypothesis, 'shapiro', self.dataset)

        assert 'p_value' in result
        assert 'statistic' in result

    def test_execute_unknown_method(self):
        """Тест выполнения неизвестного метода."""
        hypothesis = {'type': 'correlation', 'columns': ['x', 'y']}

        result = self.executor.execute(hypothesis, 'unknown_method', self.dataset)

        assert 'error' in result
        assert not result.get('is_significant', True)

    def test_execute_with_missing_data(self):
        """Тест выполнения с пропущенными данными."""
        # Создаем данные с пропусками
        df_with_nan = self.test_df.copy()
        df_with_nan.loc[0:10, 'x'] = np.nan
        dataset_with_nan = Dataset(df_with_nan)

        hypothesis = {'type': 'correlation', 'columns': ['x', 'y']}

        result = self.executor.execute(hypothesis, 'pearson', dataset_with_nan)

        # Должен обработать пропуски
        assert 'p_value' in result
        assert isinstance(result['p_value'], (int, float))

    def test_execute_small_sample(self):
        """Тест выполнения с маленькой выборкой."""
        small_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [1, 2, 3],
            'group': ['A', 'A', 'B']
        })

        small_dataset = Dataset(small_df)

        # Тест корреляции
        hypothesis_corr = {'type': 'correlation', 'columns': ['x', 'y']}
        result_corr = self.executor.execute(hypothesis_corr, 'pearson', small_dataset)

        assert 'p_value' in result_corr

        # Тест t-теста
        hypothesis_ttest = {'type': 'mean_difference', 'columns': ['x', 'group']}
        result_ttest = self.executor.execute(hypothesis_ttest, 't_test_ind', small_dataset)

        assert 'p_value' in result_ttest


class TestInterpreter:
    """Тесты интерпретатора."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = ConfigManager()
        self.interpreter = Interpreter(self.config)

    def test_interpret_significant_result(self):
        """Тест интерпретации значимого результата."""
        hypothesis = {
            'text': 'Средний возраст отличается между мужчинами и женщинами',
            'type': 'mean_difference'
        }

        analysis_result = {
            'p_value': 0.001,
            'statistic': 3.5,
            'is_significant': True,
            'group1_mean': 30.5,
            'group2_mean': 35.2,
            'effect_size': 0.8
        }

        interpretation = self.interpreter.interpret(hypothesis, analysis_result)

        assert 'conclusion' in interpretation
        assert 'explanation' in interpretation
        assert 'confidence' in interpretation

        # Проверяем, что вывод содержит ключевые слова
        conclusion = interpretation['conclusion'].lower()
        assert any(word in conclusion for word in ['подтверждена', 'значим', 'отличается'])

        # Уверенность должна быть высокой
        assert interpretation['confidence'] > 0.7

    def test_interpret_non_significant_result(self):
        """Тест интерпретации незначимого результата."""
        hypothesis = {
            'text': 'Существует корреляция между ростом и весом',
            'type': 'correlation'
        }

        analysis_result = {
            'p_value': 0.3,
            'correlation': 0.15,
            'is_significant': False
        }

        interpretation = self.interpreter.interpret(hypothesis, analysis_result)

        conclusion = interpretation['conclusion'].lower()
        assert any(word in conclusion for word in ['отклонена', 'незначим', 'не найдено'])

        # Уверенность должна быть ниже
        assert interpretation['confidence'] < 0.5

    def test_interpret_with_llm(self):
        """Тест интерпретации с LLM."""
        with patch.object(self.interpreter.llm_client, 'generate') as mock_llm:
            mock_llm.return_value = {
                'conclusion': 'Гипотеза подтверждена',
                'explanation': 'Результат статистически значим',
                'confidence': 0.9
            }

            self.interpreter.config.llm.enable = True

            hypothesis = {'text': 'Тестовая гипотеза', 'type': 'correlation'}
            analysis_result = {'p_value': 0.01, 'is_significant': True}

            interpretation = self.interpreter.interpret(hypothesis, analysis_result)

            assert interpretation['conclusion'] == 'Гипотеза подтверждена'
            assert interpretation['confidence'] == 0.9
            mock_llm.assert_called_once()

    def test_interpret_edge_cases(self):
        """Тест интерпретации граничных случаев."""
        # Граничное p-value
        hypothesis = {'text': 'Тест', 'type': 'correlation'}

        analysis_result = {
            'p_value': 0.049,  # Почти значимо
            'is_significant': True,
            'correlation': 0.2
        }

        interpretation = self.interpreter.interpret(hypothesis, analysis_result)

        # Должен быть осторожный вывод
        assert interpretation['confidence'] < 0.8

        # Очень маленькое p-value
        analysis_result['p_value'] = 0.0001
        interpretation = self.interpreter.interpret(hypothesis, analysis_result)

        assert interpretation['confidence'] > 0.9

    def test_interpret_error(self):
        """Тест интерпретации с ошибкой."""
        hypothesis = {'text': 'Тест', 'type': 'correlation'}
        analysis_result = {'error': 'Не удалось выполнить анализ'}

        interpretation = self.interpreter.interpret(hypothesis, analysis_result)

        assert 'Ошибка' in interpretation['conclusion']
        assert interpretation['confidence'] == 0.0


class TestQAInspector:
    """Тесты инспектора качества."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = ConfigManager()
        self.inspector = QAInspector(self.config)

        self.test_df = pd.DataFrame({
            'age': np.random.normal(30, 5, 100),
            'salary': np.random.normal(50000, 10000, 100),
            'department': np.random.choice(['IT', 'HR', 'Sales'], 100),
            'satisfaction': np.random.choice(['high', 'medium', 'low'], 100)
        })

        self.dataset = Dataset(self.test_df)

    def test_inspect_valid_hypothesis(self):
        """Тест проверки валидной гипотезы."""
        hypothesis = {
            'id': 1,
            'text': 'Средняя зарплата отличается по отделам',
            'type': 'mean_difference',
            'columns': ['salary', 'department']
        }

        analysis_result = {
            'p_value': 0.03,
            'is_significant': True,
            'method': 'anova'
        }

        report = self.inspector.inspect(hypothesis, 'anova', analysis_result, self.dataset)

        assert 'is_valid' in report
        assert 'score' in report
        assert 'issues' in report
        assert 'suggestions' in report

        # Валидная гипотеза должна иметь высокий счет
        assert report['score'] > 0.6

    def test_inspect_invalid_hypothesis(self):
        """Тест проверки невалидной гипотезы."""
        hypothesis = {
            'id': 1,
            'text': 'Корреляция между отделом и удовлетворенностью',
            'type': 'correlation',
            'columns': ['department', 'satisfaction']  # Обе категориальные
        }

        analysis_result = {
            'p_value': 0.5,
            'is_significant': False,
            'method': 'pearson'  # Неправильный метод
        }

        report = self.inspector.inspect(hypothesis, 'pearson', analysis_result, self.dataset)

        # Должны быть проблемы
        assert len(report['issues']) > 0
        assert report['score'] < 0.5

    def test_inspect_with_llm(self):
        """Тест проверки с LLM."""
        with patch.object(self.inspector.llm_client, 'generate') as mock_llm:
            mock_llm.return_value = {
                'score': 0.8,
                'issues': ['Небольшая выборка'],
                'suggestions': ['Увеличить размер выборки'],
                'needs_refinement': False
            }

            self.inspector.config.llm.enable = True

            hypothesis = {'text': 'Тест', 'type': 'correlation', 'columns': ['age', 'salary']}
            analysis_result = {'p_value': 0.05, 'is_significant': True}

            report = self.inspector.inspect(hypothesis, 'pearson', analysis_result, self.dataset)

            assert report['score'] == 0.8
            assert not report['needs_refinement']
            mock_llm.assert_called_once()

    def test_inspect_small_sample(self):
        """Тест проверки с маленькой выборкой."""
        small_df = pd.DataFrame({'x': [1, 2], 'y': [1, 2], 'group': ['A', 'B']})
        small_dataset = Dataset(small_df)

        hypothesis = {'text': 'Тест', 'type': 'correlation', 'columns': ['x', 'y']}
        analysis_result = {'p_value': 0.05, 'is_significant': True}

        report = self.inspector.inspect(hypothesis, 'pearson', analysis_result, small_dataset)

        # Должна быть проблема с размером выборки
        issues_text = ' '.join(report['issues']).lower()
        assert any(word in issues_text for word in ['маленьк', 'выборк', 'sample'])

    def test_inspect_multicollinearity(self):
        """Тест проверки мультиколлинеарности."""
        # Создаем сильно коррелированные переменные
        correlated_df = pd.DataFrame({
            'x1': np.arange(100),
            'x2': np.arange(100) * 1.1,  # Сильно коррелировано с x1
            'y': np.random.normal(0, 1, 100)
        })

        correlated_dataset = Dataset(correlated_df)

        hypothesis = {'text': 'Тест', 'type': 'correlation', 'columns': ['x1', 'x2']}
        analysis_result = {'p_value': 0.001, 'is_significant': True, 'correlation': 0.99}

        report = self.inspector.inspect(hypothesis, 'pearson', analysis_result, correlated_dataset)

        # Должна быть проблема с мультиколлинеарностью
        issues_text = ' '.join(report['issues']).lower()
        assert any(word in issues_text for word in ['коррел', 'multicollinear', 'зависим'])


class TestRefiner:
    """Тесты доработчика."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.config = ConfigManager()
        self.refiner = Refiner(self.config)

    def test_refine_valid_hypothesis(self):
        """Тест доработки валидной гипотезы."""
        hypothesis = {
            'id': 1,
            'text': 'Средняя зарплата отличается между IT и HR отделами',
            'type': 'mean_difference',
            'columns': ['salary', 'department']
        }

        qa_report = {
            'score': 0.85,
            'issues': [],
            'suggestions': ['Уточнить какие именно отделы сравниваются'],
            'needs_refinement': False
        }

        refined = self.refiner.refine(hypothesis, qa_report)

        # Валидная гипотеза не должна сильно меняться
        assert refined['text'] == hypothesis['text']
        assert refined['type'] == hypothesis['type']

    def test_refine_invalid_hypothesis(self):
        """Тест доработки невалидной гипотезы."""
        hypothesis = {
            'id': 1,
            'text': 'Корреляция между отделом и зарплатой',
            'type': 'correlation',
            'columns': ['department', 'salary']  # Одна категориальная
        }

        qa_report = {
            'score': 0.3,
            'issues': ['Неправильный тип гипотезы для категориальных переменных'],
            'suggestions': ['Использовать сравнение средних вместо корреляции'],
            'needs_refinement': True
        }

        refined = self.refiner.refine(hypothesis, qa_report)

        # Гипотеза должна быть улучшена
        assert refined['type'] != hypothesis['type']  # Тип должен измениться
        assert 'средн' in refined['text'].lower()  # Должно быть про средние

    def test_refine_with_llm(self):
        """Тест доработки с LLM."""
        with patch.object(self.refiner.llm_client, 'generate') as mock_llm:
            mock_llm.return_value = {
                'hypotheses': [
                    {
                        'text': 'Средняя зарплата в IT отделе выше чем в HR',
                        'type': 'mean_difference',
                        'columns': ['salary', 'department'],
                        'improvements': ['Уточнены группы сравнения']
                    }
                ]
            }

            self.refiner.config.llm.enable = True

            hypothesis = {'text': 'Зарплата отличается по отделам', 'type': 'correlation'}
            qa_report = {'needs_refinement': True, 'suggestions': ['Уточнить']}

            refined = self.refiner.refine(hypothesis, qa_report)

            assert 'IT' in refined['text']
            assert refined['type'] == 'mean_difference'
            mock_llm.assert_called_once()

    def test_refine_multiple_iterations(self):
        """Тест нескольких итераций доработки."""
        hypothesis = {
            'text': 'Связь между переменными',
            'type': 'unknown',
            'columns': []
        }

        # Симуляция нескольких отчетов QA
        qa_reports = [
            {
                'score': 0.2,
                'issues': ['Не указаны переменные'],
                'suggestions': ['Указать конкретные переменные'],
                'needs_refinement': True
            },
            {
                'score': 0.5,
                'issues': ['Неясная формулировка'],
                'suggestions': ['Сделать гипотезу более конкретной'],
                'needs_refinement': True
            },
            {
                'score': 0.8,
                'issues': [],
                'suggestions': [],
                'needs_refinement': False
            }
        ]

        for qa_report in qa_reports:
            hypothesis = self.refiner.refine(hypothesis, qa_report)

        # После доработок гипотеза должна быть лучше
        assert len(hypothesis['text']) > len('Связь между переменными')
        assert hypothesis['type'] != 'unknown'

    def test_refine_statistical_power(self):
        """Тест доработки для увеличения статистической мощности."""
        hypothesis = {
            'text': 'Маленькая разница между группами',
            'type': 'mean_difference',
            'columns': ['measurement', 'group']
        }

        qa_report = {
            'score': 0.4,
            'issues': ['Низкая статистическая мощность'],
            'suggestions': [
                'Увеличить размер выборки',
                'Использовать более чувствительный тест'
            ],
            'needs_refinement': True
        }

        refined = self.refiner.refine(hypothesis, qa_report)

        # Проверяем, что гипотеза улучшена
        assert 'мощност' not in refined['text'].lower()  # Не должно быть в тексте
        # Должна быть более конкретная формулировка


if __name__ == "__main__":
    pytest.main([__file__, "-v"])