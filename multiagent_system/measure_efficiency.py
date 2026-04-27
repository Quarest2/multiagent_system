#!/usr/bin/env python3
"""Комплексное измерение эффективности системы."""

import time
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка переменных окружения
load_dotenv()

from src.config import SystemConfig, AgentConfig, LLMConfig, DataConfig
from src.core.orchestrator import Orchestrator


class SystemEfficiencyMeasurement:
    """Измерение эффективности системы."""

    def __init__(self):
        self.metrics = {
            'quality_metrics': [],
            'performance_metrics': [],
            'ai_impact_metrics': []
        }

    def measure_all(self, datasets: list):
        """Полное измерение на нескольких датасетах."""

        print("\n" + "=" * 80)
        print("КОМПЛЕКСНОЕ ИЗМЕРЕНИЕ ЭФФЕКТИВНОСТИ СИСТЕМЫ")
        print("=" * 80)

        # Проверка API ключа
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("\n⚠️  WARNING: GROQ_API_KEY не найден в .env файле!")
            print("AI-прогоны будут идентичны базовым.\n")
        else:
            print(f"\n✓ GROQ_API_KEY найден (длина: {len(api_key)} символов)\n")

        for dataset_path in datasets:
            if not Path(dataset_path).exists():
                print(f"⚠ Пропущен: {dataset_path}")
                continue

            print(f"\n📊 Датасет: {Path(dataset_path).name}")
            print("-" * 80)

            # Измерение 1: Без AI
            print("⏳ Анализ БЕЗ AI...")
            metrics_no_ai = self._measure_single(dataset_path, use_ai=False)

            # Измерение 2: С AI
            print("\n⏳ Анализ С AI...")
            metrics_with_ai = self._measure_single(dataset_path, use_ai=True)

            # Сравнение
            comparison = self._compare_metrics(metrics_no_ai, metrics_with_ai, Path(dataset_path).name)

            self.metrics['quality_metrics'].append(comparison)

            self._print_comparison(Path(dataset_path).name, comparison)

        # Итоговая сводка
        if self.metrics['quality_metrics']:
            self._print_summary()
            self._generate_plots()

        # Сохранение
        self._save_results()

    def _measure_single(self, dataset_path: str, use_ai: bool) -> dict:
        """Измерение для одной конфигурации."""

        # Явная передача API key
        api_key = os.getenv('GROQ_API_KEY') if use_ai else None

        config = SystemConfig(
            llm=LLMConfig(
                enabled=use_ai,
                api_key=api_key
            ),
            agent=AgentConfig(
                max_hypotheses=20,
                refinement_cycles=3,  # Уменьшаем для скорости
                use_ai_generation=use_ai,
                use_ai_interpretation=use_ai,
                use_ai_qa=use_ai
            ),
            data=DataConfig()
        )

        # Запуск с замером времени
        start_time = time.time()
        orchestrator = Orchestrator(config)

        # Подавление вывода для чистоты измерений
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            results = orchestrator.run(dataset_path)
        finally:
            sys.stdout = old_stdout

        execution_time = time.time() - start_time

        if not results:
            return self._empty_metrics()

        # Сбор метрик
        significant = sum(1 for r in results if r['is_significant'])

        # Расчет interpretation_length
        interpretation_lengths = []
        for r in results:
            expl = r.get('interpretation', {}).get('explanation', '')
            interpretation_lengths.append(len(expl))

        avg_interpretation_length = np.mean(interpretation_lengths) if interpretation_lengths else 0

        # IQS расчет
        from src.utils.iqs_calculator import InsightQualityScore
        iqs = InsightQualityScore.calculate(results)

        # Подсчет типов гипотез
        hypothesis_types = {}
        for r in results:
            ht = r['hypothesis_type']
            hypothesis_types[ht] = hypothesis_types.get(ht, 0) + 1

        return {
            'execution_time': execution_time,
            'total_hypotheses': len(results),
            'significant_hypotheses': significant,
            'significance_rate': significant / len(results) * 100 if results else 0,
            'avg_quality': np.mean([r['quality_score'] for r in results]) if results else 0,
            'avg_confidence': np.mean([r['confidence'] for r in results]) if results else 0,
            'avg_p_value': np.mean([r['p_value'] for r in results if r['p_value'] > 0]) if results else 1,
            'interpretation_length': avg_interpretation_length,
            'iqs_score': iqs['total_score'],
            'iqs_grade': iqs['grade'],
            'iqs_breakdown': iqs['breakdown'],
            'hypothesis_types': hypothesis_types,
            'has_ai_explanations': any('ai_explanation' in r.get('interpretation', {}) for r in results)
        }

    def _empty_metrics(self):
        """Пустые метрики."""
        return {
            'execution_time': 0,
            'total_hypotheses': 0,
            'significant_hypotheses': 0,
            'significance_rate': 0,
            'avg_quality': 0,
            'avg_confidence': 0,
            'avg_p_value': 1,
            'interpretation_length': 0,
            'iqs_score': 0,
            'iqs_grade': 'F',
            'iqs_breakdown': {},
            'hypothesis_types': {},
            'has_ai_explanations': False
        }

    def _compare_metrics(self, no_ai: dict, with_ai: dict, dataset_name: str) -> dict:
        """Сравнение метрик."""
        return {
            'dataset_name': dataset_name,
            'time_diff': with_ai['execution_time'] - no_ai['execution_time'],
            'time_ratio': with_ai['execution_time'] / no_ai['execution_time'] if no_ai['execution_time'] > 0 else 1,
            'quality_improvement': with_ai['avg_quality'] - no_ai['avg_quality'],
            'confidence_improvement': with_ai['avg_confidence'] - no_ai['avg_confidence'],
            'significance_rate_diff': with_ai['significance_rate'] - no_ai['significance_rate'],
            'interpretation_improvement': with_ai['interpretation_length'] - no_ai['interpretation_length'],
            'iqs_improvement': with_ai['iqs_score'] - no_ai['iqs_score'],
            'ai_explanations_added': with_ai['has_ai_explanations'],
            'no_ai': no_ai,
            'with_ai': with_ai
        }

    def _print_comparison(self, dataset_name: str, comp: dict):
        """Вывод сравнения."""
        print(f"\n📈 Результаты для {dataset_name}:")
        print(f"  Время: {comp['no_ai']['execution_time']:.2f}с → {comp['with_ai']['execution_time']:.2f}с "
              f"({comp['time_diff']:+.2f}с)")

        # IQS СРАВНЕНИЕ
        print(f"  IQS: {comp['no_ai']['iqs_score']:.1f} ({comp['no_ai']['iqs_grade']}) → "
              f"{comp['with_ai']['iqs_score']:.1f} ({comp['with_ai']['iqs_grade']})")
        print(f"       Прирост: {comp['iqs_improvement']:+.1f} баллов")

        print(f"  Качество: {comp['no_ai']['avg_quality']:.3f} → {comp['with_ai']['avg_quality']:.3f} "
              f"({comp['quality_improvement']:+.3f})")
        print(f"  Уверенность: {comp['no_ai']['avg_confidence']:.3f} → {comp['with_ai']['avg_confidence']:.3f} "
              f"({comp['confidence_improvement']:+.3f})")
        print(f"  Значимых: {comp['no_ai']['significant_hypotheses']}/{comp['no_ai']['total_hypotheses']} → "
              f"{comp['with_ai']['significant_hypotheses']}/{comp['with_ai']['total_hypotheses']} "
              f"({comp['with_ai']['significance_rate']:.1f}%)")
        print(f"  Детальность: {comp['no_ai']['interpretation_length']:.0f} → "
              f"{comp['with_ai']['interpretation_length']:.0f} символов "
              f"({comp['interpretation_improvement']:+.0f})")
        print(f"  AI объяснения: {'✓ Да' if comp['ai_explanations_added'] else '✗ Нет'}")

        # Типы гипотез
        print(f"\n  Типы гипотез:")
        all_types = set(comp['no_ai']['hypothesis_types'].keys()) | set(comp['with_ai']['hypothesis_types'].keys())
        for ht in sorted(all_types):
            no_ai_count = comp['no_ai']['hypothesis_types'].get(ht, 0)
            with_ai_count = comp['with_ai']['hypothesis_types'].get(ht, 0)
            if no_ai_count > 0 or with_ai_count > 0:
                print(f"    • {ht}: {no_ai_count} → {with_ai_count}")

    def _print_summary(self):
        """Итоговая сводка."""
        if not self.metrics['quality_metrics']:
            return

        print("\n" + "=" * 80)
        print("ИТОГОВАЯ СВОДКА ПО ВСЕМ ДАТАСЕТАМ")
        print("=" * 80)

        all_comps = self.metrics['quality_metrics']

        avg_iqs_imp = np.mean([c['iqs_improvement'] for c in all_comps])
        avg_quality_imp = np.mean([c['quality_improvement'] for c in all_comps])
        avg_conf_imp = np.mean([c['confidence_improvement'] for c in all_comps])
        avg_time_overhead = np.mean([c['time_diff'] for c in all_comps])
        avg_interp_imp = np.mean([c['interpretation_improvement'] for c in all_comps])

        # Средние IQS
        avg_iqs_no_ai = np.mean([c['no_ai']['iqs_score'] for c in all_comps])
        avg_iqs_with_ai = np.mean([c['with_ai']['iqs_score'] for c in all_comps])

        print(f"\n📊 Средние показатели:")
        print(f"  IQS Score:")
        print(f"    • БЕЗ AI:  {avg_iqs_no_ai:.1f}")
        print(f"    • С AI:    {avg_iqs_with_ai:.1f}")
        print(f"    • Прирост: {avg_iqs_imp:+.1f} баллов ({avg_iqs_imp / avg_iqs_no_ai * 100:+.1f}%)")

        print(f"\n  Качество гипотез: {avg_quality_imp:+.3f} ({avg_quality_imp * 100:+.1f}%)")
        print(f"  Уверенность: {avg_conf_imp:+.3f} ({avg_conf_imp * 100:+.1f}%)")
        print(f"  Детальность интерпретаций: {avg_interp_imp:+.0f} символов")
        print(f"  Временные затраты: {avg_time_overhead:+.2f} сек")

        # AI explanations
        has_ai = sum(1 for c in all_comps if c['ai_explanations_added'])
        print(f"  AI объяснения добавлены: {has_ai}/{len(all_comps)} датасетов")

        print(f"\n✅ ВЫВОДЫ:")
        if avg_iqs_with_ai > avg_iqs_no_ai:
            print(f"  ✓ AI улучшает IQS Score ({avg_iqs_no_ai:.1f} → {avg_iqs_with_ai:.1f})")

        if avg_interp_imp > 50:
            print(f"  ✓ AI создает более детальные интерпретации (+{avg_interp_imp:.0f} символов)")

        if has_ai > 0:
            print(f"  ✓ AI добавляет практические объяснения и рекомендации")

        print(f"  • Стоимость AI: ~+{avg_time_overhead:.1f}с на анализ")

        # Детальный breakdown по датасетам
        print(f"\n📊 Детализация по датасетам:")
        for comp in all_comps:
            print(f"\n  {comp['dataset_name']}:")
            print(f"    БЕЗ AI: IQS={comp['no_ai']['iqs_score']:.1f} | "
                  f"Время={comp['no_ai']['execution_time']:.1f}с | "
                  f"Значимых={comp['no_ai']['significant_hypotheses']}/{comp['no_ai']['total_hypotheses']}")
            print(f"    С AI:   IQS={comp['with_ai']['iqs_score']:.1f} | "
                  f"Время={comp['with_ai']['execution_time']:.1f}с | "
                  f"Значимых={comp['with_ai']['significant_hypotheses']}/{comp['with_ai']['total_hypotheses']}")
            print(f"    Δ:      IQS={comp['iqs_improvement']:+.1f} | "
                  f"Время={comp['time_diff']:+.1f}с | "
                  f"Интерпретации={comp['interpretation_improvement']:+.0f} симв")

    def _save_results(self):
        """Сохранение результатов."""
        output_dir = Path('report_data')
        output_dir.mkdir(exist_ok=True)

        # Расчет summary
        avg_iqs_imp = np.mean([c['iqs_improvement'] for c in self.metrics['quality_metrics']]) if self.metrics[
            'quality_metrics'] else 0
        avg_time = np.mean([c['time_diff'] for c in self.metrics['quality_metrics']]) if self.metrics[
            'quality_metrics'] else 0

        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'datasets_analyzed': len(self.metrics['quality_metrics']),
                'avg_iqs_improvement': float(avg_iqs_imp),
                'avg_time_overhead': float(avg_time),
                'avg_iqs_no_ai': float(np.mean([c['no_ai']['iqs_score'] for c in self.metrics['quality_metrics']])) if
                self.metrics['quality_metrics'] else 0,
                'avg_iqs_with_ai': float(
                    np.mean([c['with_ai']['iqs_score'] for c in self.metrics['quality_metrics']])) if self.metrics[
                    'quality_metrics'] else 0
            },
            'metrics': self.metrics
        }

        with open('report_data/efficiency_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n💾 Метрики сохранены: report_data/efficiency_metrics.json")

    def _generate_plots(self):
        """Генерация графиков для каждого датасета отдельно."""
        if not self.metrics['quality_metrics']:
            return

        metrics = self.metrics['quality_metrics']

        for idx, m in enumerate(metrics):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'Сравнение эффективности: БЕЗ AI vs С AI\nДатасет: {m["dataset_name"]}',
                         fontsize=14, fontweight='bold')

            # 1. IQS Breakdown (С AI)
            breakdown = m['with_ai']['iqs_breakdown']
            components = ['Statistical\nQuality', 'Practical\nValue', 'Diversity', 'Novelty', 'Efficiency']
            values = [breakdown.get('statistical_quality', 0),
                      breakdown.get('practical_value', 0),
                      breakdown.get('diversity', 0),
                      breakdown.get('novelty', 0),
                      breakdown.get('efficiency', 0)]
            max_values = [30, 25, 20, 15, 10]

            y_pos = np.arange(len(components))
            axes[0].barh(y_pos, values, color='#3498DB', alpha=0.8)
            for i, (val, max_val) in enumerate(zip(values, max_values)):
                axes[0].barh(i, max_val, color='lightgray', alpha=0.3)
                axes[0].text(max_val + 1, i, f'{val:.1f}/{max_val}', va='center', fontsize=9)
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(components, fontsize=9)
            axes[0].set_xlabel('Score', fontweight='bold')
            axes[0].set_title(f'IQS Breakdown (С AI)')
            axes[0].set_xlim(0, 35)
            axes[0].grid(axis='x', alpha=0.3)

            # 2. Типы гипотез (С AI)
            types = m['with_ai']['hypothesis_types']
            colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C']
            axes[1].pie(types.values(), labels=types.keys(), autopct='%1.1f%%',
                        startangle=90, colors=colors[:len(types)], textprops={'fontsize': 9})
            axes[1].set_title('Типы гипотез (С AI)')

            plt.tight_layout()
            save_path = f'report_data/efficiency_comparison_{m["dataset_name"]}.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Дополнительно: общий график сравнения IQS и времени
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle('Сравнение эффективности по всем датасетам', fontsize=14, fontweight='bold')

        datasets = [m['dataset_name'] for m in metrics]
        no_ai_scores = [m['no_ai']['iqs_score'] for m in metrics]
        with_ai_scores = [m['with_ai']['iqs_score'] for m in metrics]
        time_no_ai = [m['no_ai']['execution_time'] for m in metrics]
        time_with_ai = [m['with_ai']['execution_time'] for m in metrics]

        x = np.arange(len(datasets))
        width = 0.35

        axes2[0].bar(x - width / 2, no_ai_scores, width, label='БЕЗ AI', color='#E74C3C', alpha=0.8)
        axes2[0].bar(x + width / 2, with_ai_scores, width, label='С AI', color='#27AE60', alpha=0.8)
        axes2[0].set_ylabel('IQS Score', fontweight='bold')
        axes2[0].set_title('IQS Score')
        axes2[0].set_xticks(x)
        axes2[0].set_xticklabels([d[:20] for d in datasets], rotation=15, ha='right')
        axes2[0].legend()
        axes2[0].grid(axis='y', alpha=0.3)

        axes2[1].bar(x - width / 2, time_no_ai, width, label='БЕЗ AI', color='#E74C3C', alpha=0.8)
        axes2[1].bar(x + width / 2, time_with_ai, width, label='С AI', color='#27AE60', alpha=0.8)
        axes2[1].set_ylabel('Время (сек)', fontweight='bold')
        axes2[1].set_title('Время выполнения')
        axes2[1].set_xticks(x)
        axes2[1].set_xticklabels([d[:20] for d in datasets], rotation=15, ha='right')
        axes2[1].legend()
        axes2[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = 'report_data/efficiency_comparison_all.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"\n📊 Графики сохранены:")
        print(f"   - По каждому датасету: report_data/efficiency_comparison_*.png")
        print(f"   - Общий: {save_path}")


if __name__ == "__main__":
    measurement = SystemEfficiencyMeasurement()

    # Датасеты для анализа
    datasets = [
        'examples/HR_comma_sep.csv'
        'examples/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        'examples/Teen_Mental_Health_Dataset.csv'
    ]

    measurement.measure_all(datasets)
