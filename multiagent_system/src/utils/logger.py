"""
Настройка логирования.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class JsonFormatter(logging.Formatter):
    """JSON форматтер для логов."""

    def format(self, record):
        log_record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)

        # Добавляем дополнительные поля
        if hasattr(record, 'extra'):
            log_record.update(record.extra)

        return json.dumps(log_record, ensure_ascii=False)


class ColorFormatter(logging.Formatter):
    """Цветной форматтер для консоли."""

    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        return f"{color}{log_message}{self.COLORS['RESET']}"


def setup_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Настройка логгера.

    Args:
        name: Имя логгера
        level: Уровень логирования
        log_file: Путь к файлу логов

    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)

    # Не создаем дублирующих обработчиков
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Создаем обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Форматтер для консоли
    console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    console_formatter = ColorFormatter(console_format)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)

    # Создаем обработчик для файла (если указан)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))

        # JSON форматтер для файла
        file_formatter = JsonFormatter()
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    # Добавляем обработчик ошибок
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(console_formatter)
    logger.addHandler(error_handler)

    # Отключаем propagation для корневого логгера
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Получение логгера по имени.

    Args:
        name: Имя логгера

    Returns:
        Логгер
    """
    return logging.getLogger(name)


class LoggerManager:
    """Менеджер логгеров системы."""

    def __init__(self, log_dir: str = "./logs", default_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.default_level = default_level
        self.loggers = {}

        # Настройка корневого логгера
        self._setup_root_logger()

    def _setup_root_logger(self):
        """Настройка корневого логгера."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)

        # Удаляем существующие обработчики
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def get_logger(self, name: str, level: Optional[str] = None,
                   log_file: Optional[str] = None) -> logging.Logger:
        """
        Получение или создание логгера.

        Args:
            name: Имя логгера
            level: Уровень логирования
            log_file: Имя файла логов (без пути)

        Returns:
            Логгер
        """
        if name in self.loggers:
            return self.loggers[name]

        # Определяем путь к файлу логов
        if log_file:
            log_file_path = self.log_dir / log_file
        else:
            log_file_path = self.log_dir / f"{name}.log"

        # Создаем логгер
        logger = setup_logger(
            name=name,
            level=level or self.default_level,
            log_file=str(log_file_path)
        )

        self.loggers[name] = logger
        return logger

    def get_all_loggers(self) -> dict:
        """Получение информации обо всех логгерах."""
        loggers_info = {}

        for name, logger in self.loggers.items():
            loggers_info[name] = {
                'level': logging.getLevelName(logger.level),
                'handlers': len(logger.handlers),
                'file_handlers': sum(1 for h in logger.handlers if isinstance(h, logging.FileHandler))
            }

        return loggers_info

    def set_level(self, name: str, level: str):
        """
        Установка уровня логирования.

        Args:
            name: Имя логгера
            level: Уровень логирования
        """
        if name in self.loggers:
            self.loggers[name].setLevel(getattr(logging, level.upper()))

            # Обновляем уровень всех обработчиков
            for handler in self.loggers[name].handlers:
                handler.setLevel(getattr(logging, level.upper()))

    def set_default_level(self, level: str):
        """
        Установка уровня логирования по умолчанию.

        Args:
            level: Уровень логирования
        """
        self.default_level = level

        # Обновляем уровень для всех логгеров
        for name in self.loggers:
            self.set_level(name, level)

    def add_file_handler(self, name: str, log_file: str, level: Optional[str] = None):
        """
        Добавление обработчика файла.

        Args:
            name: Имя логгера
            log_file: Имя файла логов
            level: Уровень логирования
        """
        if name not in self.loggers:
            return

        logger = self.loggers[name]
        log_file_path = self.log_dir / log_file

        file_handler = logging.FileHandler(str(log_file_path), encoding='utf-8')
        file_handler.setLevel(getattr(logging, (level or self.default_level).upper()))

        # JSON форматтер для файла
        file_formatter = JsonFormatter()
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    def rotate_logs(self, max_files: int = 10, max_size_mb: int = 100):
        """
        Ротация лог-файлов.

        Args:
            max_files: Максимальное количество файлов
            max_size_mb: Максимальный размер файла в МБ
        """
        import shutil

        log_files = sorted(self.log_dir.glob("*.log"))

        for log_file in log_files:
            # Проверяем размер файла
            if log_file.stat().st_size > max_size_mb * 1024 * 1024:
                # Создаем ротированный файл
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rotated_file = log_file.with_suffix(f".{timestamp}.log")
                shutil.move(log_file, rotated_file)

                # Очищаем старые ротированные файлы
                rotated_pattern = log_file.with_suffix(f".*.log")
                rotated_files = sorted(self.log_dir.glob(str(rotated_pattern.name)))

                if len(rotated_files) > max_files:
                    for old_file in rotated_files[:-max_files]:
                        old_file.unlink()

    def create_summary_report(self) -> dict:
        """Создание сводного отчета по логам."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_loggers': len(self.loggers),
            'loggers': self.get_all_loggers(),
            'log_files': []
        }

        # Информация о файлах логов
        for log_file in self.log_dir.glob("*.log"):
            try:
                file_info = {
                    'name': log_file.name,
                    'size_mb': log_file.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                }
                report['log_files'].append(file_info)
            except:
                pass

        return report