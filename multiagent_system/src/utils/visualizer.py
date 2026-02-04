"""
Визуализация результатов.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .logger import setup_logger

logger = setup_logger(__name__)


class ResultVisualizer:
    """Визуализатор результатов анализа."""

    def __init__(self, theme: str = "plotly_white"):
        """
        Инициализация визуализатора.

        Args:
            theme: Тема оформления
        """
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set3

    def create_hypothesis_quality_chart(self, results: List[Dict[str, Any]],
                                       output_path: Optional[str] = None) -> go.Figure:
        """
        Создание графика качества гипотез.

        Args:
            results: Результаты анализа
            output_path: Путь для сохранения

        Returns:
            График Plotly
        """
        if not results:
            logger.warning("Нет данных для визуализации")
            return go.Figure()

        df = pd.DataFrame(results)

        # Создаем подграфики
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Распределение p-value', 'Качество гипотез',
                          'Значимость vs Уверенность', 'Типы гипотез'),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )

        # 1. Распределение p-value
        p_values = [r.get('p_value') for r in results if r.get('p_value') is not None]

        if p_values:
            fig.add_trace(
                go.Histogram(
                    x=p_values,
                    nbinsx=20,
                    name='p-value',
                    marker_color=self.color_palette[0],
                    opacity=0.7
                ),
                row=1, col=1
            )

            # Добавляем линию значимости
            fig.add_vline(
                x=0.05, line_width=2, line_dash="dash",
                line_color="red", row=1, col=1
            )
            fig.add_annotation(
                x=0.05, y=0.95, xref="x", yref="paper",
                text="α=0.05", showarrow=False,
                font=dict(color="red"), row=1, col=1
            )

        # 2. Качество гипотез
        quality_scores = [r.get('quality_score', 0) for r in results]

        fig.add_trace(
            go.Box(
                y=quality_scores,
                name='Качество',
                marker_color=self.color_palette[1],
                boxpoints='all',
                jitter=0.3
            ),
            row=1, col=2
        )

        # 3. Значимость vs Уверенность
        fig.add_trace(
            go.Scatter(
                x=[r.get('confidence', 0) for r in results],
                y=[-np.log10(r.get('p_value', 1)) if r.get('p_value') else 0
                   for r in results],
                mode='markers',
                marker=dict(
                    size=10,
                    color=[r.get('quality_score', 0) for r in results],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Качество")
                ),
                text=[r.get('hypothesis_text', '')[:50] + '...' for r in results],
                hoverinfo='text',
                name='Гипотезы'
            ),
            row=2, col=1
        )

        # Добавляем линии для интерпретации
        fig.add_hline(y=-np.log10(0.05), line_dash="dash",
                     line_color="red", row=2, col=1)
        fig.add_vline(x=0.7, line_dash="dash",
                     line_color="green", row=2, col=1)

        # 4. Типы гипотез
        if 'hypothesis_type' in df.columns:
            type_counts = df['hypothesis_type'].value_counts()

            fig.add_trace(
                go.Bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    marker_color=self.color_palette[2],
                    name='Типы гипотез'
                ),
                row=2, col=2
            )

        # Настройка layout
        fig.update_layout(
            title_text="Анализ качества гипотез",
            showlegend=False,
            template=self.theme,
            height=800
        )

        # Настройка осей
        fig.update_xaxes(title_text="p-value", row=1, col=1)
        fig.update_yaxes(title_text="Количество", row=1, col=1)

        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text="Оценка качества", row=1, col=2)

        fig.update_xaxes(title_text="Уверенность", row=2, col=1)
        fig.update_yaxes(title_text="-log10(p-value)", row=2, col=1)

        fig.update_xaxes(title_text="Тип гипотезы", row=2, col=2)
        fig.update_yaxes(title_text="Количество", row=2, col=2)

        # Сохраняем если нужно
        if output_path:
            fig.write_html(output_path)
            logger.info(f"График сохранен: {output_path}")

        return fig

    def create_performance_dashboard(self, metrics: Dict[str, Any],
                                    output_path: Optional[str] = None) -> go.Figure:
        """
        Создание дашборда производительности.

        Args:
            metrics: Метрики системы
            output_path: Путь для сохранения

        Returns:
            Дашборд Plotly
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Время выполнения', 'Качество гипотез', 'Статистическая значимость',
                'Использование LLM', 'Кэш LLM', 'Использование ресурсов',
                'Распределение качества', 'Эффективность по типам', 'Сводные метрики'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "indicator"}],
                [{"type": "histogram"}, {"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.15
        )

        # 1. Время выполнения
        if 'time' in metrics:
            time_data = metrics['time']
            time_labels = ['Загрузка', 'Предобработка', 'Генерация', 'Анализ', 'Интерпретация']
            time_values = [
                time_data.get('data_loading_seconds', 0),
                time_data.get('preprocessing_seconds', 0),
                time_data.get('hypothesis_generation_seconds', 0),
                time_data.get('analysis_seconds', 0),
                time_data.get('interpretation_seconds', 0)
            ]

            fig.add_trace(
                go.Bar(
                    x=time_labels,
                    y=time_values,
                    marker_color=self.color_palette,
                    name='Время (сек)'
                ),
                row=1, col=1
            )

        # 2. Качество гипотез
        if 'quality' in metrics:
            quality_data = metrics['quality']

            fig.add_trace(
                go.Scatter(
                    x=['Всего', 'Значимые', 'Высокое качество'],
                    y=[
                        quality_data.get('total_hypotheses', 0),
                        quality_data.get('significant_hypotheses', 0),
                        quality_data.get('high_quality_count', 0)
                    ],
                    mode='lines+markers',
                    marker=dict(size=10, color='green'),
                    line=dict(color='green', width=2),
                    name='Качество'
                ),
                row=1, col=2
            )

        # 3. Статистическая значимость
        if 'quality' in metrics:
            significant = quality_data.get('significant_hypotheses', 0)
            non_significant = quality_data.get('total_hypotheses', 0) - significant

            fig.add_trace(
                go.Pie(
                    labels=['Значимые', 'Незначимые'],
                    values=[significant, non_significant],
                    marker_colors=['green', 'lightgray'],
                    hole=0.4,
                    name='Значимость'
                ),
                row=1, col=3
            )

        # 4. Использование LLM
        if 'llm' in metrics:
            llm_data = metrics['llm']

            fig.add_trace(
                go.Bar(
                    x=['Всего', 'Успешно', 'С ошибкой'],
                    y=[
                        llm_data.get('total_calls', 0),
                        llm_data.get('successful_calls', 0),
                        llm_data.get('failed_calls', 0)
                    ],
                    marker_color=['blue', 'green', 'red'],
                    name='LLM вызовы'
                ),
                row=2, col=1
            )

        # 5. Кэш LLM
        if 'llm' in metrics:
            hits = llm_data.get('cache_hits', 0)
            misses = llm_data.get('cache_misses', 0)

            fig.add_trace(
                go.Bar(
                    x=['Попадания', 'Промахи'],
                    y=[hits, misses],
                    marker_color=['orange', 'lightblue'],
                    name='Кэш LLM'
                ),
                row=2, col=2
            )

        # 6. Использование ресурсов
        if 'resources' in metrics:
            resources = metrics['resources']

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=resources.get('memory_usage_mb', 0),
                    title={'text': "Память (МБ)"},
                    domain={'row': 1, 'column': 0},
                    gauge={
                        'axis': {'range': [None, 1000]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 500], 'color': "lightgray"},
                            {'range': [500, 800], 'color': "gray"},
                            {'range': [800, 1000], 'color': "darkgray"}
                        ]
                    }
                ),
                row=2, col=3
            )

        # 7. Распределение качества
        if 'results' in metrics:
            quality_scores = [r.get('quality_score', 0) for r in metrics['results']]

            fig.add_trace(
                go.Histogram(
                    x=quality_scores,
                    nbinsx=10,
                    marker_color='purple',
                    opacity=0.7,
                    name='Распределение качества'
                ),
                row=3, col=1
            )

        # 8. Эффективность по типам
        if 'results' in metrics:
            df = pd.DataFrame(metrics['results'])
            if 'hypothesis_type' in df.columns:
                type_efficiency = df.groupby('hypothesis_type').agg({
                    'quality_score': 'mean',
                    'confidence': 'mean'
                }).reset_index()

                fig.add_trace(
                    go.Bar(
                        x=type_efficiency['hypothesis_type'],
                        y=type_efficiency['quality_score'],
                        marker_color='teal',
                        name='Эффективность по типам'
                    ),
                    row=3, col=2
                )

        # 9. Сводные метрики
        summary_data = [
            ['Метрика', 'Значение', 'Единица'],
            ['Общее время', f"{metrics.get('time', {}).get('total_seconds', 0):.1f}", 'сек'],
            ['Всего гипотез', str(metrics.get('quality', {}).get('total_hypotheses', 0)), 'шт'],
            ['Значимых', f"{metrics.get('quality', {}).get('significance_rate', 0):.1f}", '%'],
            ['Среднее качество', f"{metrics.get('quality', {}).get('avg_quality_score', 0):.3f}", ''],
            ['Кэш LLM', f"{metrics.get('llm', {}).get('cache_hit_rate', 0):.1f}", '%'],
            ['Память', f"{metrics.get('resources', {}).get('memory_usage_mb', 0):.1f}", 'МБ']
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=summary_data[0],
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=list(zip(*summary_data[1:])),
                    fill_color='lavender',
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=3, col=3
        )

        # Настройка layout
        fig.update_layout(
            title_text="Дашборд производительности системы",
            showlegend=False,
            template=self.theme,
            height=1200
        )

        # Сохраняем если нужно
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Дашборд сохранен: {output_path}")

        return fig

    def create_comparison_chart(self, metrics_list: List[Dict[str, Any]],
                               labels: List[str], output_path: Optional[str] = None) -> go.Figure:
        """
        Создание графика сравнения.

        Args:
            metrics_list: Список метрик для сравнения
            labels: Метки для каждого набора метрик
            output_path: Путь для сохранения

        Returns:
            График сравнения
        """
        if len(metrics_list) != len(labels):
            logger.error("Количество метрик и меток не совпадает")
            return go.Figure()

        # Подготавливаем данные для сравнения
        comparison_data = []

        for metrics, label in zip(metrics_list, labels):
            data = {
                'label': label,
                'total_time': metrics.get('time', {}).get('total_seconds', 0),
                'total_hypotheses': metrics.get('quality', {}).get('total_hypotheses', 0),
                'significance_rate': metrics.get('quality', {}).get('significance_rate', 0),
                'avg_quality': metrics.get('quality', {}).get('avg_quality_score', 0),
                'llm_hit_rate': metrics.get('llm', {}).get('cache_hit_rate', 0),
                'efficiency': 0
            }

            # Вычисляем эффективность
            if data['total_time'] > 0:
                data['efficiency'] = (
                    data['significance_rate'] * data['avg_quality'] * 100 /
                    data['total_time']
                )

            comparison_data.append(data)

        df = pd.DataFrame(comparison_data)

        # Создаем график
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Производительность', 'Качество результатов',
                'Эффективность LLM', 'Общая эффективность'
            ),
            vertical_spacing=0.15
        )

        # 1. Производительность
        fig.add_trace(
            go.Bar(
                x=df['label'],
                y=df['total_hypotheses'],
                name='Количество гипотез',
                marker_color=self.color_palette[0]
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['label'],
                y=df['total_time'],
                name='Время (сек)',
                yaxis='y2',
                line=dict(color='red', width=2),
                marker=dict(size=10, color='red')
            ),
            row=1, col=1
        )

        fig.update_layout(
            yaxis2=dict(
                title='Время (сек)',
                overlaying='y',
                side='right'
            ),
            row=1, col=1
        )

        # 2. Качество результатов
        fig.add_trace(
            go.Bar(
                x=df['label'],
                y=df['significance_rate'],
                name='Значимость (%)',
                marker_color=self.color_palette[1]
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=df['label'],
                y=df['avg_quality'],
                name='Качество',
                line=dict(color='green', width=2),
                marker=dict(size=10, color='green')
            ),
            row=1, col=2
        )

        # 3. Эффективность LLM
        fig.add_trace(
            go.Bar(
                x=df['label'],
                y=df['llm_hit_rate'],
                name='Кэш LLM (%)',
                marker_color=self.color_palette[2]
            ),
            row=2, col=1
        )

        # 4. Общая эффективность
        fig.add_trace(
            go.Bar(
                x=df['label'],
                y=df['efficiency'],
                name='Эффективность',
                marker_color=self.color_palette[3]
            ),
            row=2, col=2
        )

        # Настройка layout
        fig.update_layout(
            title_text="Сравнение производительности",
            showlegend=True,
            template=self.theme,
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Настройка осей
        fig.update_xaxes(title_text="Конфигурация", row=1, col=1)
        fig.update_yaxes(title_text="Количество гипотез", row=1, col=1)

        fig.update_xaxes(title_text="Конфигурация", row=1, col=2)
        fig.update_yaxes(title_text="Значимость (%)", row=1, col=2)

        fig.update_xaxes(title_text="Конфигурация", row=2, col=1)
        fig.update_yaxes(title_text="Кэш LLM (%)", row=2, col=1)

        fig.update_xaxes(title_text="Конфигурация", row=2, col=2)
        fig.update_yaxes(title_text="Эффективность", row=2, col=2)

        # Сохраняем если нужно
        if output_path:
            fig.write_html(output_path)
            logger.info(f"График сравнения сохранен: {output_path}")

        return fig

    def create_interactive_report(self, results: List[Dict[str, Any]],
                                 metrics: Dict[str, Any], output_path: str):
        """
        Создание интерактивного отчета.

        Args:
            results: Результаты анализа
            metrics: Метрики системы
            output_path: Путь для сохранения
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        # Создаем HTML отчет
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Отчет анализа данных</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .summary-card {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid #4CAF50;
                }}
                .summary-card h3 {{
                    margin: 0;
                    color: #333;
                    font-size: 14px;
                }}
                .summary-card .value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                    margin: 10px 0;
                }}
                .charts {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .chart-container {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .hypotheses-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                .hypotheses-table th,
                .hypotheses-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .hypotheses-table th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                .hypotheses-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .tab-container {{
                    margin-top: 30px;
                }}
                .tab {{
                    overflow: hidden;
                    border-bottom: 1px solid #ccc;
                }}
                .tab button {{
                    background-color: inherit;
                    float: left;
                    border: none;
                    outline: none;
                    cursor: pointer;
                    padding: 14px 16px;
                    transition: 0.3s;
                    font-size: 16px;
                }}
                .tab button:hover {{
                    background-color: #ddd;
                }}
                .tab button.active {{
                    background-color: #4CAF50;
                    color: white;
                }}
                .tab-content {{
                    display: none;
                    padding: 20px;
                    border-top: none;
                }}
                .tab-content.active {{
                    display: block;
                }}
                .badge {{
                    display: inline-block;
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                    margin-right: 5px;
                }}
                .badge-significant {{
                    background-color: #4CAF50;
                    color: white;
                }}
                .badge-not-significant {{
                    background-color: #f44336;
                    color: white;
                }}
                .badge-high-quality {{
                    background-color: #2196F3;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Отчет анализа данных</h1>
                    <p>Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="summary">
                    <div class="summary-card">
                        <h3>Всего гипотез</h3>
                        <div class="value">{len(results)}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Значимых гипотез</h3>
                        <div class="value">{sum(1 for r in results if r.get('is_significant', False))}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Среднее качество</h3>
                        <div class="value">{np.mean([r.get('quality_score', 0) for r in results]):.3f}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Время выполнения</h3>
                        <div class="value">{metrics.get('time', {}).get('total_seconds', 0):.1f} сек</div>
                    </div>
                </div>
                
                <div class="tab-container">
                    <div class="tab">
                        <button class="tab-button active" onclick="openTab(event, 'charts')">📈 Графики</button>
                        <button class="tab-button" onclick="openTab(event, 'hypotheses')">🔍 Гипотезы</button>
                        <button class="tab-button" onclick="openTab(event, 'metrics')">📊 Метрики</button>
                        <button class="tab-button" onclick="openTab(event, 'details')">🔧 Детали</button>
                    </div>
                    
                    <div id="charts" class="tab-content active">
                        <div id="qualityChart" class="chart-container"></div>
                        <div id="performanceChart" class="chart-container"></div>
                    </div>
                    
                    <div id="hypotheses" class="tab-content">
                        <table class="hypotheses-table">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Гипотеза</th>
                                    <th>Тип</th>
                                    <th>P-value</th>
                                    <th>Качество</th>
                                    <th>Статус</th>
                                </tr>
                            </thead>
                            <tbody>
                                {self._generate_hypotheses_table(results)}
                            </tbody>
                        </table>
                    </div>
                    
                    <div id="metrics" class="tab-content">
                        <h3>Детальные метрики</h3>
                        <pre>{json.dumps(metrics, indent=2, ensure_ascii=False)}</pre>
                    </div>
                    
                    <div id="details" class="tab-content">
                        <h3>Детали анализа</h3>
                        <div id="details-content">
                            {self._generate_details_content(results, metrics)}
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tabbuttons;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].style.display = "none";
                    }}
                    tabbuttons = document.getElementsByClassName("tab-button");
                    for (i = 0; i < tabbuttons.length; i++) {{
                        tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
                    }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
                
                // Инициализация графиков
                document.addEventListener('DOMContentLoaded', function() {{
                    // График качества
                    var qualityData = {self._get_chart_data(results)};
                    
                    // График производительности
                    var performanceData = {self._get_performance_data(metrics)};
                    
                    // Здесь можно добавить код для отрисовки графиков с помощью Plotly
                    console.log('Графики готовы к отрисовке');
                }});
            </script>
        </body>
        </html>
        """

        # Сохраняем HTML файл
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Интерактивный отчет сохранен: {output_path}")

    def _generate_hypotheses_table(self, results: List[Dict[str, Any]]) -> str:
        """Генерация HTML таблицы с гипотезами."""
        table_rows = []

        for i, result in enumerate(results[:20], 1):  # Ограничиваем 20 строками
            hypothesis_text = result.get('hypothesis_text', '')
            if len(hypothesis_text) > 80:
                hypothesis_text = hypothesis_text[:77] + '...'

            p_value = result.get('p_value', 'N/A')
            if isinstance(p_value, (int, float)):
                p_value = f"{p_value:.4f}"

            quality_score = result.get('quality_score', 0)
            quality_color = "green" if quality_score > 0.7 else "orange" if quality_score > 0.5 else "red"

            is_significant = result.get('is_significant', False)
            status_badge = ('<span class="badge badge-significant">✓ Значима</span>'
                          if is_significant else
                          '<span class="badge badge-not-significant">✗ Незначима</span>')

            if quality_score > 0.8:
                status_badge += '<span class="badge badge-high-quality">Высокое качество</span>'

            row = f"""
                <tr>
                    <td>{i}</td>
                    <td>{hypothesis_text}</td>
                    <td>{result.get('hypothesis_type', 'N/A')}</td>
                    <td>{p_value}</td>
                    <td style="color: {quality_color}; font-weight: bold;">{quality_score:.3f}</td>
                    <td>{status_badge}</td>
                </tr>
            """
            table_rows.append(row)

        return '\n'.join(table_rows)

    def _generate_details_content(self, results: List[Dict[str, Any]],
                                 metrics: Dict[str, Any]) -> str:
        """Генерация детального контента."""
        # Топ гипотезы
        top_hypotheses = sorted(results, key=lambda x: x.get('quality_score', 0), reverse=True)[:5]

        top_hypotheses_html = "<h4>Топ-5 гипотез по качеству:</h4><ol>"
        for i, hyp in enumerate(top_hypotheses, 1):
            top_hypotheses_html += f"""
                <li>
                    <strong>{hyp.get('hypothesis_text', '')}</strong><br>
                    Качество: {hyp.get('quality_score', 0):.3f}, 
                    P-value: {hyp.get('p_value', 'N/A'):.4f if isinstance(hyp.get('p_value'), (int, float)) else 'N/A'}, 
                    Метод: {hyp.get('method', 'N/A')}
                </li>
            """
        top_hypotheses_html += "</ol>"

        # Распределение по типам
        type_counts = {}
        for result in results:
            hyp_type = result.get('hypothesis_type', 'unknown')
            type_counts[hyp_type] = type_counts.get(hyp_type, 0) + 1

        type_distribution = "<h4>Распределение по типам:</h4><ul>"
        for hyp_type, count in type_counts.items():
            percentage = count / len(results) * 100
            type_distribution += f"<li>{hyp_type}: {count} ({percentage:.1f}%)</li>"
        type_distribution += "</ul>"

        # Статистика качества
        quality_scores = [r.get('quality_score', 0) for r in results]
        quality_stats = f"""
            <h4>Статистика качества:</h4>
            <ul>
                <li>Среднее: {np.mean(quality_scores):.3f}</li>
                <li>Медиана: {np.median(quality_scores):.3f}</li>
                <li>Стандартное отклонение: {np.std(quality_scores):.3f}</li>
                <li>Минимум: {np.min(quality_scores):.3f}</li>
                <li>Максимум: {np.max(quality_scores):.3f}</li>
            </ul>
        """

        return top_hypotheses_html + type_distribution + quality_stats

    def _get_chart_data(self, results: List[Dict[str, Any]]) -> str:
        """Получение данных для графика."""
        # Упрощенная реализация
        p_values = [r.get('p_value') for r in results if r.get('p_value') is not None]
        quality_scores = [r.get('quality_score', 0) for r in results]

        data = {
            'p_values': p_values[:10],  # Ограничиваем
            'quality_scores': quality_scores[:10]
        }

        return json.dumps(data)

    def _get_performance_data(self, metrics: Dict[str, Any]) -> str:
        """Получение данных производительности."""
        return json.dumps(metrics.get('time', {}))

    def save_matplotlib_figures(self, results: List[Dict[str, Any]],
                               output_dir: str = "./outputs/figures"):
        """
        Сохранение графиков matplotlib.

        Args:
            results: Результаты анализа
            output_dir: Директория для сохранения
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Стиль графиков
        plt.style.use('seaborn-v0_8-darkgrid')

        # 1. График распределения p-value
        p_values = [r.get('p_value') for r in results if r.get('p_value') is not None]

        if p_values:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.hist(p_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
            ax1.set_xlabel('p-value')
            ax1.set_ylabel('Количество')
            ax1.set_title('Распределение p-value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'p_value_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()

        # 2. График качества гипотез
        quality_scores = [r.get('quality_score', 0) for r in results]

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.boxplot(quality_scores, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen'))
        ax2.scatter(quality_scores, np.ones_like(quality_scores),
                   alpha=0.5, color='blue', s=30)
        ax2.set_xlabel('Оценка качества')
        ax2.set_title('Распределение качества гипотез')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'quality_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 3. График значимости vs уверенности
        confidences = [r.get('confidence', 0) for r in results]
        significant_flags = [r.get('is_significant', False) for r in results]

        fig3, ax3 = plt.subplots(figsize=(10, 6))

        # Разделяем на значимые и незначимые
        sig_x = [c for c, s in zip(confidences, significant_flags) if s]
        sig_y = [-np.log10(p) for p, s in zip(p_values, significant_flags) if s and p]

        non_sig_x = [c for c, s in zip(confidences, significant_flags) if not s]
        non_sig_y = [-np.log10(p) for p, s in zip(p_values, significant_flags) if not s and p]

        ax3.scatter(sig_x, sig_y, color='green', alpha=0.7, s=50,
                   label='Значимые', edgecolors='black')
        ax3.scatter(non_sig_x, non_sig_y, color='red', alpha=0.7, s=50,
                   label='Незначимые', edgecolors='black')

        ax3.axhline(y=-np.log10(0.05), color='black', linestyle='--',
                   linewidth=2, label='α=0.05')
        ax3.axvline(x=0.7, color='blue', linestyle='--',
                   linewidth=2, label='Уверенность=0.7')

        ax3.set_xlabel('Уверенность')
        ax3.set_ylabel('-log10(p-value)')
        ax3.set_title('Значимость vs Уверенность')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'significance_vs_confidence.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Графики matplotlib сохранены в: {output_dir}")

        return {
            'p_value_distribution': output_dir / 'p_value_distribution.png',
            'quality_distribution': output_dir / 'quality_distribution.png',
            'significance_vs_confidence': output_dir / 'significance_vs_confidence.png'
        }