from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="freq_methods_utils",
    version="1.0.0",
    description="Инструменты для выполнения лабораторных работ по Частотным методам ИТМО СУИР 2026",
    author="Petor Zhavoronkov",
    packages=find_packages(),
    install_requires=requirements
)