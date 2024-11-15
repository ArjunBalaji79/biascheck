from setuptools import setup, find_packages

setup(
    name="biascheck",
    version="0.1.0",
    description="An open-source library for checking bias in documents, models, datasets, and databases.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/biascheck",
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