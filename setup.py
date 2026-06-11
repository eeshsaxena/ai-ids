from setuptools import find_packages, setup

setup(
    name="ai-ids",
    version="1.0.0",
    description="AI-Based Intrusion Detection System using NSL-KDD",
    author="eeshsaxena",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "torch>=2.1.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": ["ai-ids=main:main"],
    },
)
