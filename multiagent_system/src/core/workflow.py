"""Управление процессом анализа."""

from typing import List, Dict, Any
from enum import Enum
from ..utils.logger import logger


class WorkflowStatus(Enum):
    """Статусы workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Workflow:
    """Управление workflow анализа."""

    def __init__(self):
        self.status = WorkflowStatus.PENDING
        self.steps = []
        self.current_step = 0

    def add_step(self, name: str):
        """Добавить шаг."""
        self.steps.append({'name': name, 'status': 'pending'})

    def start(self):
        """Начало работы."""
        self.status = WorkflowStatus.RUNNING
        logger.info("Workflow запущен")

    def complete_step(self, step_index: int):
        """Завершить шаг."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = 'completed'
            self.current_step = step_index + 1

    def complete(self):
        """Завершение workflow."""
        self.status = WorkflowStatus.COMPLETED
        logger.info("Workflow завершен")

    def get_status(self) -> Dict[str, Any]:
        """Получить статус."""
        return {
            'status': self.status.value,
            'current_step': self.current_step,
            'total_steps': len(self.steps),
            'steps': self.steps
        }
