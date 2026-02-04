"""
Оркестратор системы.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

from ..config import ConfigManager
from ..data.loader import DataLoader
from ..data.preprocessor import DataPreprocessor
from ..agents.hypothesis_generator import HypothesisGenerator
from ..agents.method_selector import MethodSelector
from ..agents.analysis_executor import AnalysisExecutor
from ..agents.interpreter import Interpreter
from ..agents.qa_inspector import QAInspector
from ..agents.refiner import Refiner
from ..utils.logger import setup_logger
from ..llm.prompts import PromptManager
from .workflow import Workflow, WorkflowStatus
from .evaluator import QualityEvaluator

logger = setup_logger(__name__)


@dataclass
class AnalysisResult:
    """Результат анализа гипотезы."""
    hypothesis_id: int
    hypothesis_text: str
    hypothesis_type: str
    columns_involved: List[str]
    method: str
    p_value: Optional[float]
    statistic: Optional[float]
    confidence: float
    is_significant: bool
    conclusion: str
    explanation: str
    llm_enhanced: bool = False
    quality_score: Optional[float] = None
    refinement_steps: int = 0
    errors: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hypothesis_id': self.hypothesis_id,
            'hypothesis_text': self.hypothesis_text,
            'hypothesis_type': self.hypothesis_type,
            'columns_involved': self.columns_involved,
            'method': self.method,
            'p_value': self.p_value,
            'statistic': self.statistic,
            'confidence': self.confidence,
            'is_significant': self.is_significant,
            'conclusion': self.conclusion,
            'explanation': self.explanation,
            'llm_enhanced': self.llm_enhanced,
            'quality_score': self.quality_score,
            'refinement_steps': self.refinement_steps,
            'errors': self.errors or []
        }


class Orchestrator:
    """Оркестратор мультиагентной системы."""

    def __init__(self, config: ConfigManager, verbose: bool = False):
        """
        Инициализация оркестратора.

        Args:
            config: Менеджер конфигурации
            verbose: Подробный вывод
        """
        self.config = config
        self.verbose = verbose
        self.agent_config = config.get_agent_config()
        self.data_config = config.get_data_config()

        self.data_loader = DataLoader(config)
        self.data_preprocessor = DataPreprocessor(config)
        self.hypothesis_generator = HypothesisGenerator(config)
        self.method_selector = MethodSelector(config)
        self.analysis_executor = AnalysisExecutor(config)
        self.interpreter = Interpreter(config)
        self.qa_inspector = QAInspector(config)
        self.refiner = Refiner(config)
        self.quality_evaluator = QualityEvaluator(config)
        self.prompt_manager = PromptManager(config)

        self.workflow = Workflow()
        self.results: List[AnalysisResult] = []
        self.metrics: Dict[str, Any] = {}

        logger.info("Оркестратор инициализирован")

    def run(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Основной метод запуска системы.

        Args:
            data_path: Путь к файлу с данными

        Returns:
            Список результатов анализа
        """
        try:
            self._setup_workflow()
            self.workflow.start_step(0)

            logger.info("Этап 1: Загрузка данных")
            dataset = self.data_loader.load(data_path)
            self.workflow.complete_step(0, {"rows": len(dataset.df), "columns": len(dataset.df.columns)})

            self.workflow.start_step(1)
            dataset = self.data_preprocessor.process(dataset)
            self.workflow.complete_step(1, dataset.metadata)

            logger.info("Этап 2: Генерация гипотез")
            self.workflow.start_step(2)
            hypotheses = self.hypothesis_generator.generate(dataset)

            if self.agent_config.max_hypotheses:
                hypotheses = hypotheses[:self.agent_config.max_hypotheses]

            self.workflow.complete_step(2, {"count": len(hypotheses)})

            logger.info(f"Этап 3: Обработка {len(hypotheses)} гипотез")
            self.results = []

            for i, hypothesis in enumerate(hypotheses):
                if self.verbose:
                    logger.info(f"Гипотеза {i + 1}/{len(hypotheses)}: {hypothesis['text'][:100]}...")

                try:
                    result = self._process_hypothesis(hypothesis, dataset)
                    self.results.append(result)
                except Exception as e:
                    logger.error(f"Ошибка обработки гипотезы {hypothesis['id']}: {e}")
                    continue

            self.workflow.start_step(3)
            self.workflow.complete_step(3, {"processed": len(self.results)})

            logger.info("Этап 4: Оценка качества")
            self.workflow.start_step(4)

            for result in self.results:
                result.quality_score = self.quality_evaluator.evaluate(result)

            self.workflow.complete_step(4, {"avg_quality": self._calculate_average_quality()})

            self.workflow.start_step(5)
            self.metrics = self._collect_metrics()
            self.workflow.complete_step(5, self.metrics)

            logger.info(f"Анализ завершен. Обработано: {len(self.results)}/{len(hypotheses)} гипотез")

            return [r.to_dict() for r in self.results]

        except Exception as e:
            logger.error(f"Критическая ошибка в оркестраторе: {e}", exc_info=True)
            self.workflow.fail_step(self.workflow.current_step, str(e))
            raise

    def _process_hypothesis(self, hypothesis: Dict, dataset) -> AnalysisResult:
        """
        Обработка одной гипотезы.

        Args:
            hypothesis: Гипотеза
            dataset: Обработанные данные

        Returns:
            Результат анализа
        """
        refinement_cycle = 0
        best_result = None
        best_quality = -1

        while refinement_cycle < self.agent_config.refinement_cycles:
            try:
                method = self.method_selector.select(hypothesis, dataset)

                analysis_result = self.analysis_executor.execute(hypothesis, method, dataset)

                if self.agent_config.enable_llm:
                    qa_report = self.qa_inspector.inspect(hypothesis, method, analysis_result, dataset)

                    if qa_report['needs_refinement'] and refinement_cycle < self.agent_config.refinement_cycles - 1:
                        hypothesis = self.refiner.refine(hypothesis, qa_report['feedback'])
                        refinement_cycle += 1
                        continue

                interpretation = self.interpreter.interpret(hypothesis, analysis_result)

                result = AnalysisResult(
                    hypothesis_id=hypothesis['id'],
                    hypothesis_text=hypothesis['text'],
                    hypothesis_type=hypothesis['type'],
                    columns_involved=hypothesis.get('columns', []),
                    method=method,
                    p_value=analysis_result.get('p_value'),
                    statistic=analysis_result.get('statistic'),
                    confidence=interpretation.get('confidence', 0.5),
                    is_significant=analysis_result.get('is_significant', False),
                    conclusion=interpretation.get('conclusion', 'Не удалось сделать вывод'),
                    explanation=interpretation.get('explanation', ''),
                    llm_enhanced=self.agent_config.enable_llm,
                    refinement_steps=refinement_cycle
                )

                quality = self.quality_evaluator.evaluate(result)
                if quality > best_quality:
                    best_result = result
                    best_result.quality_score = quality
                    best_quality = quality

                break

            except Exception as e:
                logger.error(f"Ошибка в цикле доработки {refinement_cycle}: {e}")
                refinement_cycle += 1

        if best_result is None:
            best_result = AnalysisResult(
                hypothesis_id=hypothesis['id'],
                hypothesis_text=hypothesis['text'],
                hypothesis_type=hypothesis['type'],
                columns_involved=hypothesis.get('columns', []),
                method='unknown',
                p_value=None,
                statistic=None,
                confidence=0.0,
                is_significant=False,
                conclusion='Не удалось проанализировать гипотезу',
                explanation=f'Ошибка: {str(e)}',
                errors=[str(e)]
            )

        return best_result

    def _setup_workflow(self):
        """Настройка workflow."""
        self.workflow.add_step("load_data", "Загрузка данных из файла")
        self.workflow.add_step("preprocess_data", "Предобработка данных")
        self.workflow.add_step("generate_hypotheses", "Генерация гипотез")
        self.workflow.add_step("process_hypotheses", "Обработка гипотез")
        self.workflow.add_step("evaluate_quality", "Оценка качества")
        self.workflow.add_step("finalize", "Финализация результатов")

    def _calculate_average_quality(self) -> float:
        """Расчет средней оценки качества."""
        if not self.results:
            return 0.0

        qualities = [r.quality_score for r in self.results if r.quality_score is not None]
        return sum(qualities) / len(qualities) if qualities else 0.0

    def _collect_metrics(self) -> Dict[str, Any]:
        """Сбор метрик системы."""
        if not self.results:
            return {}

        significant = [r for r in self.results if r.is_significant]
        high_quality = [r for r in self.results if r.quality_score and r.quality_score > 0.7]

        return {
            'total_hypotheses': len(self.results),
            'significant_hypotheses': len(significant),
            'high_quality_hypotheses': len(high_quality),
            'avg_quality_score': self._calculate_average_quality(),
            'avg_confidence': sum(r.confidence for r in self.results) / len(self.results),
            'avg_refinement_steps': sum(r.refinement_steps for r in self.results) / len(self.results),
            'llm_enhanced_count': sum(1 for r in self.results if r.llm_enhanced)
        }

    def save_results(self, results: List[Dict], output_path: str):
        """
        Сохранение результатов.

        Args:
            results: Результаты анализа
            output_path: Путь для сохранения
        """
        output_path = Path(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.time(),
                'metrics': self.metrics,
                'workflow_status': self.workflow.get_status(),
                'results': results
            }, f, ensure_ascii=False, indent=2)

        csv_path = output_path.with_suffix('.csv')
        df_results = pd.DataFrame(results)
        df_results.to_csv(csv_path, index=False, encoding='utf-8')

        logger.info(f"Результаты сохранены: {output_path}, {csv_path}")

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса системы."""
        return {
            'workflow': self.workflow.get_status(),
            'metrics': self.metrics,
            'results_count': len(self.results)
        }