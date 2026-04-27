"""Главный файл с поддержкой AI."""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.config import SystemConfig, AgentConfig, DataConfig, LLMConfig
from src.core.orchestrator import Orchestrator
from src.utils.logger import setup_logger

# Загружаем переменные окружения
load_dotenv()

logger = setup_logger(__name__)


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description='Мультиагентная система с AI для генерации и проверки гипотез'
    )
    
    parser.add_argument('data_path', type=str, help='Путь к данным (CSV/Excel)')
    parser.add_argument('--max-hypotheses', type=int, default=20)
    parser.add_argument('--refinement-cycles', type=int, default=3)
    parser.add_argument('--significance', type=float, default=0.05)
    parser.add_argument('--output', type=str, default='report_data/output.json')
    parser.add_argument('--no-ai', action='store_true', help='Отключить AI')
    
    args = parser.parse_args()
    
    if not Path(args.data_path).exists():
        logger.error(f"Файл не найден: {args.data_path}")
        sys.exit(1)
    
    # Конфигурация
    config = SystemConfig(
        llm=LLMConfig(enabled=not args.no_ai),
        agent=AgentConfig(
            max_hypotheses=args.max_hypotheses,
            refinement_cycles=args.refinement_cycles,
            significance_level=args.significance
        ),
        data=DataConfig()
    )
    
    print("\n" + "=" * 60)
    print("🤖 МУЛЬТИАГЕНТНАЯ СИСТЕМА С AI")
    print("=" * 60)
    print(f"Данные: {args.data_path}")
    print(f"AI-агенты: {'✓ Включены' if config.llm.enabled else '✗ Отключены'}")
    print(f"Макс. гипотез: {config.agent.max_hypotheses}")
    print(f"Циклов доработки: {config.agent.refinement_cycles}")
    print("=" * 60 + "\n")
    
    try:
        orchestrator = Orchestrator(config)
        results = orchestrator.run(args.data_path)
        orchestrator.save_results(args.output)
        
        print("\n" + "=" * 60)
        print("✓ АНАЛИЗ ЗАВЕРШЕН")
        print("=" * 60)
        print(f"Результаты: {args.output}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
