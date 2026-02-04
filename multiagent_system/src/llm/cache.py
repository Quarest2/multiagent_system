"""
Кэширование LLM запросов.
"""

import json
import hashlib
import time
from typing import Any, Dict, Optional, Union
from pathlib import Path
import pickle
import gzip
import sqlite3
from datetime import datetime, timedelta

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMCache:
    """Кэш для LLM запросов."""

    def __init__(self, cache_dir: str = "./cache", max_size_mb: int = 1000, ttl_hours: int = 24):
        """
        Инициализация кэша.

        Args:
            cache_dir: Директория для кэша
            max_size_mb: Максимальный размер кэша в МБ
            ttl_hours: Время жизни записей в часах
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_hours * 3600

        # Инициализация БД SQLite
        self.db_path = self.cache_dir / "llm_cache.db"
        self._init_database()

        logger.info(f"Кэш LLM инициализирован: {self.cache_dir}")

    def _init_database(self):
        """Инициализация базы данных."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Создание таблицы кэша
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                prompt_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Создание индексов
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache(last_accessed)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model ON cache(model)')

        conn.commit()
        conn.close()

    def _generate_key(self, prompt: str, model: str, temperature: float) -> str:
        """
        Генерация ключа кэша.

        Args:
            prompt: Текст промпта
            model: Название модели
            temperature: Температура генерации

        Returns:
            Уникальный ключ
        """
        # Создаем хэш из промпта, модели и температуры
        content = f"{prompt}|{model}|{temperature:.2f}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_prompt_hash(self, prompt: str) -> str:
        """Получение хэша промпта."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def get(self, prompt: str, model: str, temperature: float) -> Optional[Dict[str, Any]]:
        """
        Получение ответа из кэша.

        Args:
            prompt: Текст промпта
            model: Название модели
            temperature: Температура генерации

        Returns:
            Ответ из кэша или None
        """
        key = self._generate_key(prompt, model, temperature)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(
                'SELECT * FROM cache WHERE key = ?',
                (key,)
            )

            row = cursor.fetchone()

            if row:
                # Проверяем TTL
                created_at = datetime.fromisoformat(row['created_at'])
                if (datetime.now() - created_at).total_seconds() > self.ttl_seconds:
                    # Удаляем просроченную запись
                    cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
                    conn.commit()
                    logger.debug(f"Удалена просроченная запись кэша: {key}")
                    return None

                # Обновляем счетчик обращений
                cursor.execute(
                    'UPDATE cache SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE key = ?',
                    (key,)
                )
                conn.commit()

                # Десериализуем ответ
                response = json.loads(row['response'])

                logger.debug(f"Кэш попадание: {key[:16]}... (обращений: {row['access_count'] + 1})")
                return response
            else:
                logger.debug(f"Кэш промах: {key[:16]}...")
                return None

        except Exception as e:
            logger.error(f"Ошибка при чтении из кэша: {e}")
            return None
        finally:
            conn.close()

    def set(self, prompt: str, model: str, temperature: float, response: Dict[str, Any]):
        """
        Сохранение ответа в кэш.

        Args:
            prompt: Текст промпта
            model: Название модели
            temperature: Температура генерации
            response: Ответ LLM
        """
        key = self._generate_key(prompt, model, temperature)
        prompt_hash = self._get_prompt_hash(prompt)

        # Сериализуем ответ
        response_json = json.dumps(response, ensure_ascii=False)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Проверяем размер кэша
            cursor.execute('SELECT COUNT(*) as count, SUM(LENGTH(response)) as size FROM cache')
            result = cursor.fetchone()

            if result and result[1]:
                cache_size_mb = result[1] / (1024 * 1024)
                if cache_size_mb > (self.max_size_bytes / (1024 * 1024)):
                    # Удаляем старые записи
                    self._cleanup_old_entries(cursor)

            # Сохраняем или обновляем запись
            cursor.execute('''
                INSERT OR REPLACE INTO cache (key, prompt_hash, model, response, created_at, access_count)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 0)
            ''', (key, prompt_hash, model, response_json))

            conn.commit()
            logger.debug(f"Сохранено в кэш: {key[:16]}...")

        except Exception as e:
            logger.error(f"Ошибка при сохранении в кэш: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _cleanup_old_entries(self, cursor: sqlite3.Cursor):
        """Очистка старых записей кэша."""
        try:
            # Удаляем 20% самых старых и редко используемых записей
            cursor.execute('''
                DELETE FROM cache 
                WHERE key IN (
                    SELECT key FROM cache 
                    ORDER BY last_accessed ASC, access_count ASC 
                    LIMIT (SELECT COUNT(*) * 0.2 FROM cache)
                )
            ''')

            deleted_count = cursor.rowcount
            logger.info(f"Очистка кэша: удалено {deleted_count} старых записей")

        except Exception as e:
            logger.error(f"Ошибка при очистке кэша: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Общая статистика
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(LENGTH(response)) as total_size_bytes,
                    AVG(access_count) as avg_access_count,
                    MIN(created_at) as oldest_entry,
                    MAX(created_at) as newest_entry
                FROM cache
            ''')
            stats_row = cursor.fetchone()

            # Распределение по моделям
            cursor.execute('''
                SELECT model, COUNT(*) as count, AVG(access_count) as avg_access
                FROM cache 
                GROUP BY model 
                ORDER BY count DESC
            ''')
            models = cursor.fetchall()

            # Хиты/промахи (приблизительно)
            cursor.execute('SELECT COUNT(*) as hits FROM cache WHERE access_count > 0')
            hits = cursor.fetchone()[0]

            total = stats_row[0] if stats_row and stats_row[0] else 0
            hit_rate = hits / total if total > 0 else 0

            return {
                'total_entries': total,
                'total_size_mb': (stats_row[1] or 0) / (1024 * 1024),
                'avg_access_count': stats_row[2] or 0,
                'oldest_entry': stats_row[3],
                'newest_entry': stats_row[4],
                'hit_rate': hit_rate,
                'models': [
                    {
                        'model': model[0],
                        'count': model[1],
                        'avg_access': model[2]
                    }
                    for model in models
                ]
            }

        except Exception as e:
            logger.error(f"Ошибка при получении статистики кэша: {e}")
            return {}
        finally:
            conn.close()

    def clear(self):
        """Очистка всего кэша."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('DELETE FROM cache')
            conn.commit()
            logger.info("Кэш полностью очищен")

            # Оптимизация БД
            cursor.execute('VACUUM')
            conn.commit()

        except Exception as e:
            logger.error(f"Ошибка при очистке кэша: {e}")
            conn.rollback()
        finally:
            conn.close()

    def clear_expired(self):
        """Очистка просроченных записей."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                DELETE FROM cache 
                WHERE julianday('now') - julianday(created_at) > ?
            ''', (self.ttl_seconds / 86400,))

            deleted_count = cursor.rowcount
            conn.commit()

            if deleted_count > 0:
                logger.info(f"Удалено {deleted_count} просроченных записей кэша")

        except Exception as e:
            logger.error(f"Ошибка при очистке просроченных записей: {e}")
            conn.rollback()
        finally:
            conn.close()

    def search(self, query: str, limit: int = 10) -> list:
        """
        Поиск в кэше по промпту.

        Args:
            query: Строка поиска
            limit: Максимальное количество результатов

        Returns:
            Список найденных записей
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT key, model, created_at, access_count, last_accessed
                FROM cache 
                WHERE prompt_hash IN (
                    SELECT prompt_hash FROM cache 
                    WHERE key IN (
                        SELECT key FROM cache 
                        WHERE response LIKE ?
                    )
                )
                ORDER BY last_accessed DESC 
                LIMIT ?
            ''', (f'%{query}%', limit))

            results = []
            for row in cursor.fetchall():
                results.append(dict(row))

            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске в кэше: {e}")
            return []
        finally:
            conn.close()

    def backup(self, backup_dir: str = "./cache/backups"):
        """
        Создание резервной копии кэша.

        Args:
            backup_dir: Директория для бэкапов
        """
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"llm_cache_backup_{timestamp}.db"

        try:
            # Копируем файл БД
            import shutil
            shutil.copy2(self.db_path, backup_path)

            # Сжимаем бэкап
            compressed_path = backup_path.with_suffix('.db.gz')
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            # Удаляем несжатый файл
            backup_path.unlink()

            logger.info(f"Резервная копия создана: {compressed_path}")

            # Очистка старых бэкапов (оставляем последние 10)
            backup_files = sorted(backup_dir.glob("*.db.gz"))
            if len(backup_files) > 10:
                for old_backup in backup_files[:-10]:
                    old_backup.unlink()
                    logger.debug(f"Удален старый бэкап: {old_backup}")

            return str(compressed_path)

        except Exception as e:
            logger.error(f"Ошибка при создании резервной копии: {e}")
            return None


class MemoryCache:
    """Простой in-memory кэш для тестирования."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times = {}

    def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        if key in self.cache:
            # Проверяем TTL
            if time.time() - self.access_times.get(key, 0) > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                return None

            # Обновляем время доступа
            self.access_times[key] = time.time()
            return self.cache[key]

        return None

    def set(self, key: str, value: Any):
        """Сохранение значения в кэш."""
        # Проверяем размер кэша
        if len(self.cache) >= self.max_size:
            # Удаляем самую старую запись
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = value
        self.access_times[key] = time.time()

    def clear(self):
        """Очистка кэша."""
        self.cache.clear()
        self.access_times.clear()