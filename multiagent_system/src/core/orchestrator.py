"""Оркестратор с AI-агентами."""

import time
import json
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

from ..config import SystemConfig
from ..data.loader import DataLoader, Dataset
from ..agents.hypothesis_generator import HypothesisGenerator
from ..agents.method_selector import MethodSelector
from ..agents.analysis_executor import AnalysisExecutor
from ..agents.qa_inspector import QAInspector
from ..agents.refiner import Refiner
from ..agents.interpreter import Interpreter
from ..core.workflow import Workflow
from ..core.evaluator import QualityEvaluator
from ..utils.logger import logger
from ..utils.metrics import SystemMetrics
from ..utils.iqs_calculator import InsightQualityScore


def convert_to_python_types(obj):
    """Конвертация numpy типов."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    return obj


@dataclass
class AnalysisResult:
    """Результат анализа."""
    hypothesis_id: int
    hypothesis_text: str
    hypothesis_type: str
    method: str
    p_value: float
    statistic: float
    is_significant: bool
    quality_score: float
    confidence: float
    interpretation: Dict[str, Any]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return convert_to_python_types({
            'hypothesis_id': self.hypothesis_id,
            'hypothesis_text': self.hypothesis_text,
            'hypothesis_type': self.hypothesis_type,
            'method': self.method,
            'p_value': self.p_value,
            'statistic': self.statistic,
            'is_significant': self.is_significant,
            'quality_score': round(self.quality_score, 3),
            'confidence': round(self.confidence, 3),
            'interpretation': self.interpretation,
            'details': self.details
        })


class Orchestrator:
    """Оркестратор с AI-агентами."""

    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig.default()

        # Инициализация всех агентов
        self.data_loader = DataLoader()
        self.hypothesis_generator = HypothesisGenerator(self.config.agent, self.config.llm)
        self.method_selector = MethodSelector(self.config.agent)
        self.analysis_executor = AnalysisExecutor(self.config.agent)
        self.qa_inspector = QAInspector(self.config.agent, self.config.llm)
        self.refiner = Refiner(self.config.agent)
        self.interpreter = Interpreter(self.config.agent, self.config.llm)
        self.quality_evaluator = QualityEvaluator(self.config.agent)

        self.workflow = Workflow()
        self.results: List[AnalysisResult] = []

        # Проверка AI
        if self.config.llm.enabled:
            from ..llm.groq_client import GroqLLMClient
            test_client = GroqLLMClient(self.config.llm.api_key)
            if test_client.is_available():
                logger.info("✓ AI-агенты активированы (Groq LLM)")
            else:
                logger.warning("⚠ AI-агенты недоступны. Работа в базовом режиме.")

        logger.info("Оркестратор инициализирован")

    def run(self, data_path: str) -> List[Dict[str, Any]]:
        """Запуск анализа."""
        start_time = time.time()

        self._setup_workflow()
        self.workflow.start()

        try:
            # Шаг 1: Загрузка
            logger.info("=" * 60)
            logger.info("ШАГ 1: ЗАГРУЗКА ДАННЫХ")
            logger.info("=" * 60)
            dataset = self.data_loader.load(data_path)
            print(dataset.get_summary())
            self.workflow.complete_step(0)

            # Шаг 2: Генерация гипотез
            logger.info("\n" + "=" * 60)
            logger.info("ШАГ 2: ГЕНЕРАЦИЯ ГИПОТЕЗ (С AI)")
            logger.info("=" * 60)
            hypotheses = self.hypothesis_generator.generate(dataset)
            print(f"\nСгенерировано гипотез: {len(hypotheses)}")
            self._print_hypotheses(hypotheses)
            self.workflow.complete_step(1)

            # Шаг 3: Анализ
            logger.info("\n" + "=" * 60)
            logger.info("ШАГ 3: АНАЛИЗ ГИПОТЕЗ")
            logger.info("=" * 60)
            self.results = []

            for i, hypothesis in enumerate(hypotheses, 1):
                print(f"\n[{i}/{len(hypotheses)}] Гипотеза #{hypothesis['id']}: {hypothesis['text'][:60]}...")
                result = self._process_hypothesis(hypothesis, dataset)
                if result:
                    self.results.append(result)

            self.workflow.complete_step(2)

            # Шаг 4: Итоги
            logger.info("\n" + "=" * 60)
            logger.info("ШАГ 4: РЕЗУЛЬТАТЫ")
            logger.info("=" * 60)
            metrics = self._collect_metrics(time.time() - start_time)
            self._print_results(metrics)
            self.workflow.complete_step(3)

            self.workflow.complete()

            return [r.to_dict() for r in self.results]

        except Exception as e:
            logger.error(f"Ошибка: {e}", exc_info=True)
            raise

    def _setup_workflow(self):
        """Настройка workflow."""
        self.workflow.add_step("Загрузка")
        self.workflow.add_step("Генерация")
        self.workflow.add_step("Анализ")
        self.workflow.add_step("Финализация")

    def _process_hypothesis(self, hypothesis: Dict, dataset: Dataset) -> AnalysisResult:
        """Обработка гипотезы с AI."""
        best_result = None
        best_quality = -1

        for cycle in range(self.config.agent.refinement_cycles):
            try:
                # Выбор метода
                method = self.method_selector.select(hypothesis, dataset)

                if method == 'unknown':
                    logger.warning("  ⚠ Метод не выбран")
                    continue

                # Анализ
                analysis_result = self.analysis_executor.execute(hypothesis, method, dataset)

                if 'error' in analysis_result:
                    logger.warning(f"  ⚠ {analysis_result['error']}")
                    continue

                # QA проверка (с AI)
                qa_report = self.qa_inspector.inspect(hypothesis, method, analysis_result, dataset)

                # Интерпретация (с AI)
                interpretation = self.interpreter.interpret(hypothesis, analysis_result)

                # Оценка качества
                quality_score = self.quality_evaluator.evaluate(analysis_result)

                result = AnalysisResult(
                    hypothesis_id=hypothesis['id'],
                    hypothesis_text=hypothesis['text'],
                    hypothesis_type=hypothesis['type'],
                    method=method,
                    p_value=analysis_result.get('p_value', -1),
                    statistic=analysis_result.get('statistic', 0),
                    is_significant=analysis_result.get('is_significant', False),
                    quality_score=quality_score,
                    confidence=self._calculate_confidence(analysis_result, qa_report),
                    interpretation=interpretation,
                    details=analysis_result
                )

                if quality_score > best_quality:
                    best_result = result
                    best_quality = quality_score

                print(f"  Метод: {method}, p-value: {result.p_value:.4f}, "
                      f"Качество: {quality_score:.2f}")

                if not qa_report.get('needs_refinement', False):
                    break

                if cycle < self.config.agent.refinement_cycles - 1:
                    print(f"  ⟳ Доработка {cycle + 1}")
                    hypothesis = self.refiner.refine(hypothesis, qa_report)

            except Exception as e:
                logger.error(f"  ✗ Ошибка: {e}")
                continue

        if best_result:
            status = "✓ ЗНАЧИМА" if best_result.is_significant else "✗ НЕ ЗНАЧИМА"
            print(f"  → {status}")

        return best_result

    def _calculate_confidence(self, result: Dict, qa_report: Dict) -> float:
        """Расчет уверенности."""
        confidence = 0.5

        p_value = result.get('p_value', 1.0)
        if p_value < 0.001:
            confidence += 0.3
        elif p_value < 0.01:
            confidence += 0.2
        elif p_value < 0.05:
            confidence += 0.1

        quality = qa_report.get('quality_score', 0)
        confidence += quality * 0.2

        issues_count = len(qa_report.get('issues', []))
        confidence -= issues_count * 0.05

        return max(0.0, min(confidence, 1.0))

    def _collect_metrics(self, processing_time: float) -> Dict[str, Any]:
        """Сбор метрик."""
        if not self.results:
            return {
                'system_metrics': SystemMetrics().to_dict(),
                'iqs': InsightQualityScore.calculate([])
            }

        # Базовые метрики
        significant_count = sum(1 for r in self.results if r.is_significant)
        avg_quality = sum(r.quality_score for r in self.results) / len(self.results)
        avg_confidence = sum(r.confidence for r in self.results) / len(self.results)

        system_metrics = SystemMetrics(
            total_hypotheses=len(self.results),
            significant_hypotheses=significant_count,
            avg_quality_score=avg_quality,
            avg_confidence=avg_confidence,
            processing_time=processing_time
        )

        # IQS метрика
        results_dicts = [r.to_dict() for r in self.results]
        iqs = InsightQualityScore.calculate(results_dicts)

        return {
            'system_metrics': system_metrics.to_dict(),
            'iqs': iqs
        }

    def _print_hypotheses(self, hypotheses: List[Dict]):
        """Вывод гипотез."""
        ai_count = sum(1 for h in hypotheses if h.get('source') == 'ai')
        if ai_count > 0:
            print(f"  (в том числе {ai_count} от AI)")

        for h in hypotheses[:5]:
            source_mark = "🤖" if h.get('source') == 'ai' else "📊"
            print(f"  {source_mark} {h['text']}")
        if len(hypotheses) > 5:
            print(f"  ... и еще {len(hypotheses) - 5}")

    def _print_results(self, metrics: Dict):
        """Вывод результатов."""
        print("\n" + "=" * 60)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 60)

        # ИСПРАВЛЕНО: metrics уже словарь
        m = metrics['system_metrics']
        print(f"Всего гипотез:        {m['total_hypotheses']}")
        print(f"Значимых:             {m['significant_hypotheses']} ({m['significance_rate']:.1f}%)")
        print(f"Среднее качество:     {m['avg_quality_score']:.3f}")
        print(f"Средняя уверенность:  {m['avg_confidence']:.3f}")
        print(f"Время:                {m['processing_time']:.2f} сек")

        # IQS МЕТРИКА
        iqs = metrics['iqs']
        print("\n" + "=" * 60)
        print("📊 INSIGHT QUALITY SCORE (IQS)")
        print("=" * 60)
        print(f"Общий балл:           {iqs['total_score']:.1f}/100")
        print(f"Оценка:               {iqs['grade']}")
        print(f"\nДетализация:")
        for component, score in iqs['breakdown'].items():
            component_name = {
                'statistical_quality': 'Статистическое качество',
                'practical_value': 'Практическая ценность',
                'diversity': 'Разнообразие',
                'novelty': 'Новизна',
                'efficiency': 'Эффективность'
            }.get(component, component)
            print(f"  • {component_name}: {score:.1f}/30" if component == 'statistical_quality'
                  else f"  • {component_name}: {score:.1f}")

        print(f"\n💡 Рекомендации:")
        for rec in iqs['recommendations']:
            print(f"  {rec}")

        # Топ гипотезы
        print("\n" + "=" * 60)
        print("ТОП-3 РЕЗУЛЬТАТА")
        print("=" * 60)

        significant = [r for r in self.results if r.is_significant]
        significant.sort(key=lambda x: x.confidence, reverse=True)

        for i, r in enumerate(significant[:3], 1):
            print(f"\n{i}. {r.hypothesis_text}")
            print(f"   Тип: {r.hypothesis_type} | Метод: {r.method}")
            print(f"   P-value: {r.p_value:.4f} | Качество: {r.quality_score:.2f}")

            if 'explanation' in r.interpretation:
                expl = r.interpretation['explanation']
                print(f"   💡 {expl[:150]}...")

    def save_results(self, output_path: str):
        """Сохранение результатов."""
        metrics = self._collect_metrics(0)  # Без времени для сохранения

        results_dict = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'max_hypotheses': self.config.agent.max_hypotheses,
                'refinement_cycles': self.config.agent.refinement_cycles,
                'ai_enabled': self.config.llm.enabled
            },
            'results': [r.to_dict() for r in self.results],
            'metrics': convert_to_python_types(metrics['system_metrics']),
            'iqs': metrics['iqs']
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Результаты: {output_path}")

        df = pd.DataFrame([r.to_dict() for r in self.results])
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"✓ CSV: {csv_path}")
