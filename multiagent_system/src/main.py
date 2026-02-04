"""
Точка входа в систему.
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

from src.core.orchestrator import Orchestrator
from src.utils.logger import setup_logger
from src.config import ConfigManager

logger = setup_logger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """Настройка парсера аргументов."""
    parser = argparse.ArgumentParser(
        description="Мультиагентная система генерации и проверки гипотез"
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Путь к файлу с данными (CSV, Excel, JSON)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Путь к файлу конфигурации'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь для сохранения результатов'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Подробный вывод'
    )

    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Отключить использование LLM'
    )

    parser.add_argument(
        '--max-hypotheses',
        type=int,
        default=None,
        help='Максимальное количество гипотез'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Создать визуализации'
    )

    return parser


def main():
    """Основная функция."""
    parser = setup_argparser()
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Файл не найден: {data_path}")
        return 1

    config_manager = ConfigManager(args.config)

    if args.no_llm:
        config_manager.update_llm_config(enable=False)

    if args.max_hypotheses:
        config_manager.update_agent_config(max_hypotheses=args.max_hypotheses)

    orchestrator = Orchestrator(
        config=config_manager,
        verbose=args.verbose
    )

    try:
        logger.info(f"Начало анализа данных из: {data_path}")

        results = orchestrator.run(str(data_path))

        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(config_manager.config.output_dir) / f"results_{timestamp}.json"

        orchestrator.save_results(results, str(output_path))

        if args.visualize:
            from src.utils.visualizer import ResultVisualizer
            visualizer = ResultVisualizer()
            vis_path = output_path.with_suffix('.html')
            visualizer.create_dashboard(results, str(vis_path))
            logger.info(f"Дашборд сохранен: {vis_path}")

        print("\n" + "=" * 70)
        print("АНАЛИЗ ЗАВЕРШЕН")
        print("=" * 70)

        significant = [r for r in results if r.get('is_significant', False)]
        llm_enhanced = [r for r in results if r.get('llm_enhanced', False)]

        print(f"Всего гипотез: {len(results)}")
        print(f"Значимых гипотез: {len(significant)} ({len(significant) / len(results) * 100:.1f}%)")
        print(f"Усилено LLM: {len(llm_enhanced)}")
        print(f"Средняя уверенность: {sum(r.get('confidence', 0) for r in results) / len(results):.2f}")
        print(f"Результаты сохранены: {output_path}")

        if results:
            print("\n" + "=" * 70)
            print("ТОП-3 ГИПОТЕЗЫ")
            print("=" * 70)

            sorted_results = sorted(results, key=lambda x: x.get('confidence', 0), reverse=True)
            for i, result in enumerate(sorted_results[:3], 1):
                print(f"\n{i}. {result['hypothesis_text']}")
                print(f"   Метод: {result.get('method', 'N/A')}")
                print(f"   P-value: {result.get('p_value', 'N/A'):.4f}")
                print(f"   Заключение: {result.get('conclusion', 'N/A')}")
                print(f"   Уверенность: {result.get('confidence', 0):.2f}")

        return 0

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())