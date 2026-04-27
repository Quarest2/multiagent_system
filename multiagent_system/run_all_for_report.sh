#!/bin/bash
echo "🚀 Запуск полного анализа для отчета..."

# 1. Основной анализ
echo "1/4 Основной анализ данных..."
python main.py examples/HR_comma_sep.csv --output report_data/HR_final.json
python main.py examples/WA_Fn-UseC_-Telco-Customer-Churn.csv --output report_data/WA_Fn-UseC_-Telco-Customer-Churn_final.json
python main.py examples/Teen_Mental_Health_Dataset.csv --output report_data/Teen_Mental_Health_Dataset_final.json

# 2. Измерение эффективности
echo "2/4 Измерение эффективности..."
python measure_efficiency.py

# 4. Упаковка для отчета
echo "4/4 Упаковка результатов..."
mkdir -p report_data
cp report_data/*.json results/*.png report_data/*.csv report_data/ 2>/dev/null
zip -r report_data.zip report_data/

echo ""
echo "✅ ГОТОВО!"
echo "Файлы для отчета: report_data.zip"
ls -lh report_data.zip
