from setuptools import setup, find_packages

import subprocess
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)

setup(
    name="biascheck",
    version="0.2.0",
    author="Arjun Balaji",
    author_email="",
    description="A library for detecting and analyzing bias in text, datasets, and language models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/biascheck",
    packages=find_packages(),
    python_requires=">=3.9,<3.11",
    install_requires=[
        "numpy"
        "torch",
        "transformers",
        "langchain",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "PyPDF2",
        "textblob",
        "wordcloud",
        "faiss-cpu",
        "sentence-transformers",
        "PyMuPDF",
        "spacy",
        "scipy",
        "langchain-community",
        "seaborn",
        
    ],

    entry_points={
        "console_scripts": [
            "biascheck=biascheck.cli:main",
        ],
    },

)