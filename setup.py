from setuptools import setup, find_packages

setup(
    name="biascheck",
    version="0.1.0",
    description="An open-source library for checking bias in documents, models, datasets, and databases.",
    author="Arjun Balaji",
    author_email="arjun.balaji.02@example.com",
    url="https://github.com/ArjunBalaji79/biascheck",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "faiss-cpu",
        "spacy",
        "pandas",
        "matplotlib",
        "wordcloud",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "biascheck=cli:cli",
        ]
    },
)