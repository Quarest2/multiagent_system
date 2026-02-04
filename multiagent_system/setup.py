"""
Setup файл для установки пакета.
"""

from setuptools import setup, find_packages
import os
import re

# Чтение README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Чтение версии из __init__.py
with open(os.path.join("src", "__init__.py"), "r", encoding="utf-8") as f:
    version_match = re.search(r'^__version__ = ["\']([^"\']*)["\']', f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        version = "1.0.0"

# Чтение зависимостей
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Дополнительные зависимости для разработки
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
    "jupyter>=1.0.0",
    "notebook>=7.0.0",
]

# Дополнительные зависимости для LLM
llm_requirements = [
    "openai>=1.0.0",
    "anthropic>=0.8.0",
    "google-generativeai>=0.3.0",
    "langchain>=0.1.0",
    "langchain-community>=0.0.10",
]

setup(
    name="hypothesis-multiagent-system",
    version=version,
    author="Research Team",
    author_email="research@example.com",
    description="Мультиагентная система для автоматической генерации и проверки статистических гипотез на основе табличных данных",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hypothesis-multiagent-system",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "llm": llm_requirements,
        "all": dev_requirements + llm_requirements,
    },
    entry_points={
        "console_scripts": [
            "hypothesis-system=src.main:main",
            "hypothesis-demo=src.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "config/*.yaml",
            "config/*.yml",
            "examples/*.json",
            "examples/*.csv",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hypothesis-multiagent-system/issues",
        "Source": "https://github.com/yourusername/hypothesis-multiagent-system",
        "Documentation": "https://github.com/yourusername/hypothesis-multiagent-system/wiki",
    },
    keywords=[
        "data-science",
        "machine-learning",
        "statistics",
        "hypothesis-testing",
        "multi-agent",
        "automation",
        "artificial-intelligence",
        "data-analysis",
    ],
    license="MIT",
)

# Инструкции для установки
print("\n" + "="*70)
print("Установка мультиагентной системы генерации гипотез")
print("="*70)
print("\nДля установки базовой версии:")
print("  pip install .")
print("\nДля установки с LLM поддержкой:")
print("  pip install .[llm]")
print("\nДля установки для разработки:")
print("  pip install .[all]")
print("\nДля запуска системы:")
print("  hypothesis-system --data path/to/your/data.csv")
print("\nДля запуска демо:")
print("  hypothesis-demo")
print("="*70)