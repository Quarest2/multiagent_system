#!/usr/bin/env python3
"""Генерация графиков для отчета."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Загрузка данных
with open('report_data/efficiency_metrics.json', 'r') as f:
    data = json.load(f)

metrics = data['metrics']['quality_metrics']

# График 1: IQS Score сравнение
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. IQS Comparison
datasets = []
no_ai_scores = []
with_ai_scores = []

for m in metrics:
    datasets.append(m['no_ai'].get('dataset_name', 'Dataset'))
    no_ai_scores.append(m['no_ai']['iqs_score'])
    with_ai_scores.append(m['with_ai']['iqs_score'])

x = np.arange(len(datasets))
width = 0.35

axes[0, 0].bar(x - width / 2, no_ai_scores, width, label='БЕЗ AI', color='#E74C3C')
axes[0, 0].bar(x + width / 2, with_ai_scores, width, label='С AI', color='#27AE60')
axes[0, 0].set_ylabel('IQS Score')
axes[0, 0].set_title('Сравнение IQS Score: БЕЗ AI vs С AI')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(datasets, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].axhline(y=65, color='gray', linestyle='--', alpha=0.5, label='Grade B')

# 2. Breakdown по компонентам (для первого датасета)
if metrics:
    breakdown = metrics[0]['with_ai']['iqs_breakdown']
    components = list(breakdown.keys())
    values = list(breakdown.values())

    axes[0, 1].barh(components, values, color='#3498DB')
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_title('IQS Breakdown (С AI)')
    axes[0, 1].set_xlim(0, 30)

# 3. Время выполнения
time_no_ai = [m['no_ai']['execution_time'] for m in metrics]
time_with_ai = [m['with_ai']['execution_time'] for m in metrics]

axes[1, 0].bar(x - width / 2, time_no_ai, width, label='БЕЗ AI', color='#E74C3C')
axes[1, 0].bar(x + width / 2, time_with_ai, width, label='С AI', color='#27AE60')
axes[1, 0].set_ylabel('Время (сек)')
axes[1, 0].set_title('Время выполнения')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(datasets, rotation=45, ha='right')
axes[1, 0].legend()

# 4. Типы гипотез (для первого датасета)
if metrics and 'hypothesis_types' in metrics[0]['with_ai']:
    types = metrics[0]['with_ai']['hypothesis_types']

    axes[1, 1].pie(types.values(), labels=types.keys(), autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Распределение типов гипотез (С AI)')

plt.tight_layout()
plt.savefig('report_data/efficiency_comparison.png', dpi=300, bbox_inches='tight')
print("✓ График сохранен: report_data/efficiency_comparison.png")

plt.show()
