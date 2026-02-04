"""
Управление workflow системы.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import time
import json
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class WorkflowStatus(Enum):
    """Статусы workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Шаг workflow."""
    name: str
    description: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def start(self):
        """Начало выполнения шага."""
        self.status = WorkflowStatus.RUNNING
        self.start_time = time.time()
        logger.info(f"Начало шага: {self.name}")

    def complete(self, result: Any = None):
        """Успешное завершение шага."""
        self.status = WorkflowStatus.COMPLETED
        self.end_time = time.time()
        self.duration = self.end_time - (self.start_time or self.end_time)
        self.result = result
        logger.info(f"Завершен шаг: {self.name} (за {self.duration:.2f} сек)")

    def fail(self, error: str):
        """Завершение шага с ошибкой."""
        self.status = WorkflowStatus.FAILED
        self.end_time = time.time()
        self.duration = self.end_time - (self.start_time or self.end_time)
        self.error = error
        logger.error(f"Ошибка шага {self.name}: {error}")

    def skip(self):
        """Пропуск шага."""
        self.status = WorkflowStatus.SKIPPED
        logger.info(f"Пропущен шаг: {self.name}")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'has_result': self.result is not None,
            'has_error': self.error is not None,
            'metadata': self.metadata
        }


class Workflow:
    """Управление workflow мультиагентной системы."""

    def __init__(self, workflow_id: Optional[str] = None):
        """
        Инициализация workflow.

        Args:
            workflow_id: Уникальный идентификатор workflow
        """
        self.workflow_id = workflow_id or f"workflow_{int(time.time())}"
        self.steps: List[WorkflowStep] = []
        self.current_step_index: int = -1
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.metadata: Dict[str, Any] = {}

        logger.info(f"Создан workflow: {self.workflow_id}")

    def add_step(self, name: str, description: str, metadata: Optional[Dict] = None) -> int:
        """
        Добавление шага в workflow.

        Args:
            name: Название шага
            description: Описание шага
            metadata: Дополнительные метаданные

        Returns:
            Индекс добавленного шага
        """
        step = WorkflowStep(name=name, description=description)
        if metadata:
            step.metadata.update(metadata)

        self.steps.append(step)
        return len(self.steps) - 1

    def start(self):
        """Начало выполнения workflow."""
        self.start_time = time.time()
        logger.info(f"Начало workflow: {self.workflow_id}")

    def complete(self):
        """Успешное завершение workflow."""
        self.end_time = time.time()
        logger.info(f"Завершение workflow: {self.workflow_id}")

    def fail(self, error: str):
        """Завершение workflow с ошибкой."""
        self.end_time = time.time()
        if self.current_step_index >= 0 and self.current_step_index < len(self.steps):
            self.steps[self.current_step_index].fail(error)
        logger.error(f"Ошибка workflow {self.workflow_id}: {error}")

    def start_step(self, step_index: int):
        """
        Начало выполнения шага.

        Args:
            step_index: Индекс шага
        """
        if 0 <= step_index < len(self.steps):
            self.current_step_index = step_index
            self.steps[step_index].start()
        else:
            logger.warning(f"Неверный индекс шага: {step_index}")

    def complete_step(self, step_index: int, result: Any = None):
        """
        Успешное завершение шага.

        Args:
            step_index: Индекс шага
            result: Результат шага
        """
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].complete(result)
        else:
            logger.warning(f"Неверный индекс шага: {step_index}")

    def fail_step(self, step_index: int, error: str):
        """
        Завершение шага с ошибкой.

        Args:
            step_index: Индекс шага
            error: Сообщение об ошибке
        """
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].fail(error)
        else:
            logger.warning(f"Неверный индекс шага: {step_index}")

    def skip_step(self, step_index: int):
        """
        Пропуск шага.

        Args:
            step_index: Индекс шага
        """
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].skip()
        else:
            logger.warning(f"Неверный индекс шага: {step_index}")

    def get_current_step(self) -> Optional[WorkflowStep]:
        """Получение текущего шага."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса workflow."""
        completed_steps = sum(1 for s in self.steps if s.status == WorkflowStatus.COMPLETED)
        failed_steps = sum(1 for s in self.steps if s.status == WorkflowStatus.FAILED)
        running_steps = sum(1 for s in self.steps if s.status == WorkflowStatus.RUNNING)

        total_duration = 0.0
        if self.end_time and self.start_time:
            total_duration = self.end_time - self.start_time
        elif self.start_time:
            total_duration = time.time() - self.start_time

        progress = 0.0
        if self.steps:
            progress = completed_steps / len(self.steps)

        return {
            'workflow_id': self.workflow_id,
            'total_steps': len(self.steps),
            'completed_steps': completed_steps,
            'failed_steps': failed_steps,
            'running_steps': running_steps,
            'progress': progress,
            'total_duration': total_duration,
            'current_step': self.current_step_index,
            'current_step_name': self.get_current_step().name if self.get_current_step() else None,
            'status': self._get_overall_status().value,
            'start_time': self.start_time,
            'end_time': self.end_time
        }

    def _get_overall_status(self) -> WorkflowStatus:
        """Определение общего статуса workflow."""
        if any(s.status == WorkflowStatus.FAILED for s in self.steps):
            return WorkflowStatus.FAILED

        if all(s.status == WorkflowStatus.COMPLETED for s in self.steps):
            return WorkflowStatus.COMPLETED

        if any(s.status == WorkflowStatus.RUNNING for s in self.steps):
            return WorkflowStatus.RUNNING

        if any(s.status == WorkflowStatus.PENDING for s in self.steps):
            return WorkflowStatus.PENDING

        return WorkflowStatus.COMPLETED

    def get_step_results(self) -> List[Dict[str, Any]]:
        """Получение результатов всех шагов."""
        return [step.to_dict() for step in self.steps]

    def add_metadata(self, key: str, value: Any):
        """Добавление метаданных."""
        self.metadata[key] = value

    def save_to_file(self, filepath: str):
        """
        Сохранение workflow в файл.

        Args:
            filepath: Путь к файлу
        """
        data = {
            'workflow_id': self.workflow_id,
            'metadata': self.metadata,
            'status': self.get_status(),
            'steps': self.get_step_results(),
            'timestamp': time.time()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Workflow сохранен: {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> "Workflow":
        """
        Загрузка workflow из файла.

        Args:
            filepath: Путь к файлу

        Returns:
            Загруженный workflow
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        workflow = cls(workflow_id=data['workflow_id'])
        workflow.metadata = data.get('metadata', {})

        # Восстанавливаем шаги (только структуру, без состояния)
        for step_data in data.get('steps', []):
            step = WorkflowStep(
                name=step_data['name'],
                description=step_data['description']
            )
            workflow.steps.append(step)

        logger.info(f"Workflow загружен: {filepath}")
        return workflow

    def create_checkpoint(self, checkpoint_dir: str) -> str:
        """
        Создание контрольной точки.

        Args:
            checkpoint_dir: Директория для контрольных точек

        Returns:
            Путь к файлу контрольной точки
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        checkpoint_file = checkpoint_dir / f"checkpoint_{self.workflow_id}_{timestamp}.json"

        self.save_to_file(str(checkpoint_file))
        return str(checkpoint_file)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        completed_steps = [s for s in self.steps if s.status == WorkflowStatus.COMPLETED]

        if not completed_steps:
            return {}

        durations = [s.duration for s in completed_steps if s.duration is not None]

        return {
            'total_completed': len(completed_steps),
            'total_duration': sum(durations) if durations else 0,
            'avg_step_duration': sum(durations) / len(durations) if durations else 0,
            'min_step_duration': min(durations) if durations else 0,
            'max_step_duration': max(durations) if durations else 0,
            'throughput': len(completed_steps) / (sum(durations) if durations else 1)
        }