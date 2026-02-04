"""
Сбор метрик системы.
"""

import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from .logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TimeMetrics:
    """Метрики времени."""
    total_seconds: float = 0.0
    data_loading_seconds: float = 0.0
    preprocessing_seconds: float = 0.0
    hypothesis_generation_seconds: float = 0.0
    analysis_seconds: float = 0.0
    interpretation_seconds: float = 0.0
    llm_calls_seconds: float = 0.0
    per_hypothesis_avg_seconds: float = 0.0


@dataclass
class QualityMetrics:
    """Метрики качества."""
    total_hypotheses: int = 0
    significant_hypotheses: int = 0
    significance_rate: float = 0.0
    avg_p_value: Optional[float] = None
    avg_confidence: float = 0.0
    avg_quality_score: float = 0.0
    high_quality_count: int = 0
    llm_enhanced_count: int = 0
    error_rate: float = 0.0


@dataclass
class LLMMetrics:
    """Метрики LLM."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    avg_response_time: float = 0.0
    total_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0


@dataclass
class ResourceMetrics:
    """Метрики ресурсов."""
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    disk_usage_mb: float = 0.0


@dataclass
class SystemMetrics:
    """Сводные метрики системы."""
    timestamp: str
    workflow_id: str
    dataset_name: str
    time: TimeMetrics
    quality: QualityMetrics
    llm: LLMMetrics
    resources: ResourceMetrics
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        result = {
            'timestamp': self.timestamp,
            'workflow_id': self.workflow_id,
            'dataset_name': self.dataset_name,
            'time': asdict(self.time),
            'quality': asdict(self.quality),
            'llm': asdict(self.llm),
            'resources': asdict(self.resources),
            'metadata': self.metadata
        }
        return result

    def to_json(self) -> str:
        """Преобразование в JSON."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save_to_file(self, filepath: str):
        """
        Сохранение в файл.

        Args:
            filepath: Путь к файлу
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Метрики сохранены: {filepath}")


class MetricsCollector:
    """Сборщик метрик системы."""

    def __init__(self, workflow_id: str):
        """
        Инициализация сборщика метрик.

        Args:
            workflow_id: Идентификатор workflow
        """
        self.workflow_id = workflow_id
        self.start_time = time.time()

        # Инициализация метрик
        self.time_metrics = TimeMetrics()
        self.quality_metrics = QualityMetrics()
        self.llm_metrics = LLMMetrics()
        self.resource_metrics = ResourceMetrics()

        # Текущие измерения
        self.current_stage = None
        self.stage_start_time = None
        self.hypothesis_times = []
        self.llm_response_times = []
        self.llm_tokens = []

        # Счетчики
        self.hypothesis_count = 0
        self.significant_count = 0
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"Сборщик метрик инициализирован для workflow: {workflow_id}")

    def start_stage(self, stage_name: str):
        """
        Начало этапа.

        Args:
            stage_name: Название этапа
        """
        if self.current_stage:
            self.end_stage()

        self.current_stage = stage_name
        self.stage_start_time = time.time()
        logger.debug(f"Начало этапа: {stage_name}")

    def end_stage(self):
        """Завершение текущего этапа."""
        if self.current_stage and self.stage_start_time:
            stage_duration = time.time() - self.stage_start_time

            # Обновляем метрики времени
            if self.current_stage == 'data_loading':
                self.time_metrics.data_loading_seconds = stage_duration
            elif self.current_stage == 'preprocessing':
                self.time_metrics.preprocessing_seconds = stage_duration
            elif self.current_stage == 'hypothesis_generation':
                self.time_metrics.hypothesis_generation_seconds = stage_duration
            elif self.current_stage == 'analysis':
                self.time_metrics.analysis_seconds = stage_duration
            elif self.current_stage == 'interpretation':
                self.time_metrics.interpretation_seconds = stage_duration
            elif self.current_stage == 'llm_calls':
                self.time_metrics.llm_calls_seconds = stage_duration

            logger.debug(f"Завершение этапа {self.current_stage}: {stage_duration:.2f} сек")

            self.current_stage = None
            self.stage_start_time = None

    def record_hypothesis_analysis(self, duration: float, is_significant: bool,
                                  p_value: Optional[float], confidence: float):
        """
        Запись метрик анализа гипотезы.

        Args:
            duration: Время анализа
            is_significant: Статистическая значимость
            p_value: P-value
            confidence: Уверенность
        """
        self.hypothesis_count += 1
        self.hypothesis_times.append(duration)

        if is_significant:
            self.significant_count += 1

        # Обновляем агрегированные метрики
        self.quality_metrics.total_hypotheses = self.hypothesis_count
        self.quality_metrics.significant_hypotheses = self.significant_count

        if self.hypothesis_count > 0:
            self.quality_metrics.significance_rate = (
                self.significant_count / self.hypothesis_count * 100
            )

    def record_llm_call(self, duration: float, success: bool, tokens: int = 0,
                       cache_hit: bool = False):
        """
        Запись метрик вызова LLM.

        Args:
            duration: Время ответа
            success: Успешность вызова
            tokens: Количество токенов
            cache_hit: Попадание в кэш
        """
        self.llm_call_count += 1
        self.llm_response_times.append(duration)
        self.llm_tokens.append(tokens)

        if success:
            self.llm_success_count += 1

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        # Обновляем агрегированные метрики
        self.llm_metrics.total_calls = self.llm_call_count
        self.llm_metrics.successful_calls = self.llm_success_count
        self.llm_metrics.failed_calls = self.llm_call_count - self.llm_success_count

        if self.llm_response_times:
            self.llm_metrics.avg_response_time = np.mean(self.llm_response_times)

        if self.llm_tokens:
            self.llm_metrics.total_tokens = sum(self.llm_tokens)

        self.llm_metrics.cache_hits = self.cache_hits
        self.llm_metrics.cache_misses = self.cache_misses

        if self.llm_call_count > 0:
            self.llm_metrics.cache_hit_rate = (
                self.cache_hits / self.llm_call_count * 100
            )

    def record_error(self):
        """Запись ошибки."""
        self.error_count += 1

    def update_quality_scores(self, quality_scores: List[float], confidences: List[float],
                             p_values: List[Optional[float]], llm_enhanced_count: int):
        """
        Обновление метрик качества.

        Args:
            quality_scores: Оценки качества
            confidences: Уверенности
            p_values: P-values
            llm_enhanced_count: Количество улучшенных LLM гипотез
        """
        if quality_scores:
            self.quality_metrics.avg_quality_score = np.mean(quality_scores)
            self.quality_metrics.high_quality_count = sum(1 for s in quality_scores if s > 0.7)

        if confidences:
            self.quality_metrics.avg_confidence = np.mean(confidences)

        # Фильтруем None значения
        valid_p_values = [p for p in p_values if p is not None]
        if valid_p_values:
            self.quality_metrics.avg_p_value = np.mean(valid_p_values)

        self.quality_metrics.llm_enhanced_count = llm_enhanced_count

        if self.hypothesis_count > 0:
            self.quality_metrics.error_rate = self.error_count / self.hypothesis_count * 100

    def update_resource_metrics(self):
        """Обновление метрик ресурсов."""
        try:
            import psutil

            # Использование памяти
            process = psutil.Process()
            memory_info = process.memory_info()
            self.resource_metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)

            # Использование CPU
            self.resource_metrics.cpu_percent = process.cpu_percent(interval=0.1)

            # Использование диска (для текущей директории)
            disk_usage = psutil.disk_usage('.')
            self.resource_metrics.disk_usage_mb = disk_usage.used / (1024 * 1024)

        except ImportError:
            logger.warning("psutil не установлен, метрики ресурсов недоступны")
        except Exception as e:
            logger.error(f"Ошибка при обновлении метрик ресурсов: {e}")

    def get_metrics(self, dataset_name: str = "unknown") -> SystemMetrics:
        """
        Получение сводных метрик.

        Args:
            dataset_name: Название датасета

        Returns:
            Сводные метрики системы
        """
        # Завершаем текущий этап
        if self.current_stage:
            self.end_stage()

        # Обновляем общее время
        total_time = time.time() - self.start_time
        self.time_metrics.total_seconds = total_time

        # Среднее время на гипотезу
        if self.hypothesis_times:
            self.time_metrics.per_hypothesis_avg_seconds = np.mean(self.hypothesis_times)

        # Обновляем метрики ресурсов
        self.update_resource_metrics()

        # Создаем метаданные
        metadata = {
            'hypothesis_count': self.hypothesis_count,
            'significant_count': self.significant_count,
            'llm_call_count': self.llm_call_count,
            'error_count': self.error_count,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }

        # Создаем сводные метрики
        system_metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            workflow_id=self.workflow_id,
            dataset_name=dataset_name,
            time=self.time_metrics,
            quality=self.quality_metrics,
            llm=self.llm_metrics,
            resources=self.resource_metrics,
            metadata=metadata
        )

        return system_metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """Получение сводки производительности."""
        metrics = self.get_metrics()

        summary = {
            'workflow_id': metrics.workflow_id,
            'dataset': metrics.dataset_name,
            'total_time_seconds': metrics.time.total_seconds,
            'hypotheses_per_second': (
                metrics.quality.total_hypotheses / metrics.time.total_seconds
                if metrics.time.total_seconds > 0 else 0
            ),
            'significance_rate': metrics.quality.significance_rate,
            'avg_confidence': metrics.quality.avg_confidence,
            'avg_quality_score': metrics.quality.avg_quality_score,
            'llm_cache_hit_rate': metrics.llm.cache_hit_rate,
            'efficiency_score': self._calculate_efficiency_score(metrics)
        }

        return summary

    def _calculate_efficiency_score(self, metrics: SystemMetrics) -> float:
        """Вычисление оценки эффективности."""
        if metrics.quality.total_hypotheses == 0:
            return 0.0

        # Взвешенная оценка эффективности
        scores = []
        weights = []

        # 1. Скорость обработки (30%)
        if metrics.time.total_seconds > 0:
            speed_score = min(
                metrics.quality.total_hypotheses / metrics.time.total_seconds * 10,
                1.0
            )
            scores.append(speed_score)
            weights.append(0.3)

        # 2. Качество результатов (40%)
        quality_score = metrics.quality.avg_quality_score
        scores.append(quality_score)
        weights.append(0.4)

        # 3. Эффективность LLM (20%)
        llm_efficiency = metrics.llm.cache_hit_rate / 100  # Нормализуем к 0-1
        scores.append(llm_efficiency)
        weights.append(0.2)

        # 4. Надежность (10%)
        reliability = 1 - (metrics.quality.error_rate / 100)
        scores.append(reliability)
        weights.append(0.1)

        # Взвешенная сумма
        total_score = sum(s * w for s, w in zip(scores, weights))
        return min(max(total_score, 0.0), 1.0)

    def save_report(self, output_dir: str = "./outputs/metrics"):
        """
        Сохранение отчета с метриками.

        Args:
            output_dir: Директория для сохранения
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Получаем метрики
        metrics = self.get_metrics()

        # Сохраняем JSON
        json_path = output_dir / f"metrics_{self.workflow_id}.json"
        metrics.save_to_file(str(json_path))

        # Создаем CSV отчет
        csv_path = output_dir / f"metrics_summary_{self.workflow_id}.csv"
        self._create_csv_report(metrics, str(csv_path))

        # Создаем текстовый отчет
        txt_path = output_dir / f"metrics_report_{self.workflow_id}.txt"
        self._create_text_report(metrics, str(txt_path))

        logger.info(f"Отчеты с метриками сохранены: {json_path}, {csv_path}, {txt_path}")

        return {
            'json': str(json_path),
            'csv': str(csv_path),
            'txt': str(txt_path)
        }

    def _create_csv_report(self, metrics: SystemMetrics, filepath: str):
        """Создание CSV отчета."""
        # Подготавливаем данные
        data = {
            'timestamp': [metrics.timestamp],
            'workflow_id': [metrics.workflow_id],
            'dataset': [metrics.dataset_name],
            'total_time_seconds': [metrics.time.total_seconds],
            'total_hypotheses': [metrics.quality.total_hypotheses],
            'significant_hypotheses': [metrics.quality.significant_hypotheses],
            'significance_rate': [metrics.quality.significance_rate],
            'avg_quality_score': [metrics.quality.avg_quality_score],
            'avg_confidence': [metrics.quality.avg_confidence],
            'llm_total_calls': [metrics.llm.total_calls],
            'llm_cache_hit_rate': [metrics.llm.cache_hit_rate],
            'memory_usage_mb': [metrics.resources.memory_usage_mb],
            'cpu_percent': [metrics.resources.cpu_percent]
        }

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')

    def _create_text_report(self, metrics: SystemMetrics, filepath: str):
        """Создание текстового отчета."""
        report_lines = [
            "=" * 70,
            "ОТЧЕТ О РАБОТЕ СИСТЕМЫ ГЕНЕРАЦИИ ГИПОТЕЗ",
            "=" * 70,
            f"Workflow ID: {metrics.workflow_id}",
            f"Датасет: {metrics.dataset_name}",
            f"Время выполнения: {metrics.timestamp}",
            "",
            "МЕТРИКИ ВРЕМЕНИ:",
            "-" * 40,
            f"Общее время: {metrics.time.total_seconds:.2f} сек",
            f"Загрузка данных: {metrics.time.data_loading_seconds:.2f} сек",
            f"Предобработка: {metrics.time.preprocessing_seconds:.2f} сек",
            f"Генерация гипотез: {metrics.time.hypothesis_generation_seconds:.2f} сек",
            f"Анализ гипотез: {metrics.time.analysis_seconds:.2f} сек",
            f"Интерпретация: {metrics.time.interpretation_seconds:.2f} сек",
            f"Вызовы LLM: {metrics.time.llm_calls_seconds:.2f} сек",
            f"Среднее время на гипотезу: {metrics.time.per_hypothesis_avg_seconds:.2f} сек",
            "",
            "МЕТРИКИ КАЧЕСТВА:",
            "-" * 40,
            f"Всего гипотез: {metrics.quality.total_hypotheses}",
            f"Значимых гипотез: {metrics.quality.significant_hypotheses}",
            f"Процент значимых: {metrics.quality.significance_rate:.1f}%",
            f"Среднее качество: {metrics.quality.avg_quality_score:.3f}",
            f"Высококачественных: {metrics.quality.high_quality_count}",
            f"Средняя уверенность: {metrics.quality.avg_confidence:.3f}",
            f"Улучшено LLM: {metrics.quality.llm_enhanced_count}",
            f"Процент ошибок: {metrics.quality.error_rate:.1f}%",
            "",
            "МЕТРИКИ LLM:",
            "-" * 40,
            f"Всего вызовов: {metrics.llm.total_calls}",
            f"Успешных вызовов: {metrics.llm.successful_calls}",
            f"Неудачных вызовов: {metrics.llm.failed_calls}",
            f"Среднее время ответа: {metrics.llm.avg_response_time:.2f} сек",
            f"Всего токенов: {metrics.llm.total_tokens}",
            f"Попаданий в кэш: {metrics.llm.cache_hits}",
            f"Промахов кэша: {metrics.llm.cache_misses}",
            f"Процент попаданий: {metrics.llm.cache_hit_rate:.1f}%",
            "",
            "МЕТРИКИ РЕСУРСОВ:",
            "-" * 40,
            f"Использование памяти: {metrics.resources.memory_usage_mb:.1f} МБ",
            f"Использование CPU: {metrics.resources.cpu_percent:.1f}%",
            f"Использование диска: {metrics.resources.disk_usage_mb:.1f} МБ",
            "",
            "СВОДКА ПРОИЗВОДИТЕЛЬНОСТИ:",
            "-" * 40,
            f"Гипотез в секунду: {metrics.quality.total_hypotheses / metrics.time.total_seconds:.2f}",
            f"Оценка эффективности: {self._calculate_efficiency_score(metrics):.3f}",
            "=" * 70
        ]

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

    @classmethod
    def compare_metrics(cls, metrics_list: List[SystemMetrics]) -> pd.DataFrame:
        """
        Сравнение метрик разных прогонов.

        Args:
            metrics_list: Список метрик

        Returns:
            DataFrame с сравнением
        """
        if not metrics_list:
            return pd.DataFrame()

        comparison_data = []

        for metrics in metrics_list:
            row = {
                'workflow_id': metrics.workflow_id,
                'dataset': metrics.dataset_name,
                'total_time': metrics.time.total_seconds,
                'total_hypotheses': metrics.quality.total_hypotheses,
                'significance_rate': metrics.quality.significance_rate,
                'avg_quality': metrics.quality.avg_quality_score,
                'llm_hit_rate': metrics.llm.cache_hit_rate,
                'memory_mb': metrics.resources.memory_usage_mb
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Добавляем относительные метрики
        if len(df) > 1:
            df['time_per_hypothesis'] = df['total_time'] / df['total_hypotheses']
            df['efficiency'] = df['significance_rate'] * df['avg_quality'] / df['time_per_hypothesis']

            # Нормализуем эффективность
            if df['efficiency'].max() > 0:
                df['efficiency_normalized'] = df['efficiency'] / df['efficiency'].max() * 100

        return df