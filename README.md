# **BiasCheck: An Open-Source Library for Bias Detection**

BiasCheck is a robust and modular Python library designed to analyze and detect bias in text, models, and datasets. It provides tools for researchers, data scientists, and developers to measure various forms of bias (e.g., stereotypical, cultural) and assess the quality of language model outputs or textual data.

---

## **Features**
- **Modular Design**: BiasCheck offers modular and extensible classes for different bias analysis tasks.
- **Bias Detection**: Analyze text, datasets, language models or databases for various types of bias.
- **Support for RAG**: Automatically create Retrieval-Augmented Generation (RAG) pipelines using documents or PDFs.
- **Sentiment Analysis**: Assess sentiment polarity alongside bias.
- **Visualization**: Visualize flagged sentences and bias types in your analysis.

---

## **Main Classes**

### **1. `DocuCheck`**
Analyze bias in standalone text documents or files.

#### Key Features:
- Accepts text data or documents (e.g., PDF, TXT).
- Detects flagged sentences and calculates a bias score.
- Optionally uses a list of polarizing terms for context-specific bias detection.

#### Example:
```python
from biascheck.analysis.docucheck import DocuCheck

data = "This is a sample document that may contain biases."
terms = ["biased", "lazy", "discrimination"]

analyzer = DocuCheck(data=data, terms=terms)
result = analyzer.analyze(verbose=False)
print(result)
```

### **2. SetCheck**

Analyze entire datasets (e.g., DataFrames) for skewed or biased records.

#### Key Features:
- Works with Python DataFrames and CSV files.
- Adds bias-related columns to the dataset.
- Returns flagged records and overall bias analysis.

#### Example:
```python
from biascheck.analysis.setcheck import SetCheck

data = [{"text": "A biased example."}, {"text": "A neutral sentence."}]
terms = ["bias", "stereotype"]

analyzer = SetCheck(data=data, input_cols=["text"], terms=terms)
flagged_df = analyzer.analyze(top_n=5)
print(flagged_df)
```

### **3. ModuCheck**

Analyze bias in language model outputs and create RAG pipelines.

#### Key Features:
- Supports HuggingFace, Ollama, and local GGUF models.
- Detects bias in generated outputs based on user-provided topics.
- Automatically builds a RAG pipeline if a document is provided.
- Saves flagged outputs and bias results to a DataFrame.

#### Example:
```python
from biascheck.analysis.moducheck import ModuCheck
from langchain.llms import Ollama

model = Ollama(model="llama3")
topics = ["The role of women in leadership", "Cultural diversity"]

analyzer = ModuCheck(model=model, terms=["bias", "stereotype"], document="file.pdf")
result = analyzer.analyze(topics=topics, num_responses=5)
print(result)
```

### **4. Visualiser**

Visualize the results of bias analysis.

#### Key Features:
- Generates bar charts for flagged bias categories.
- Visualizes flagged sentences and bias distribution.

#### Example:
```python
from biascheck.visualisation.visualiser import Visualiser

visualiser = Visualiser()
visualiser.plot_bias_categories(flagged_records)
```

## **Setup Instructions**

### Prerequisites
- Python 3.8 or above.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/biascheck.git
cd biascheck
```

2. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # For Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the library:
```bash
pip install -e .
```

## **Usage**

### Run Examples
The notebooks/ directory contains example scripts for all analysis classes:
```bash
python notebooks/moducheck_example.py
python notebooks/docucheck_example.py
```

## **Contributing**

We welcome contributions! Please fork the repository, make your changes, and submit a pull request. Ensure all new features are covered with appropriate tests.

## **Future Work**
- Multimodal Support: Expand the library to include image, video, and audio bias detection.
- Enhanced RAG Pipelines: Improve integration with custom retrievers.
- Advanced Bias Categories: Expand predefined bias categories for deeper contextual analysis.

## **Contact**

For questions, suggestions, or feedback, reach out to the project maintainer:
- Name: Arjun Balaji
- Email: arjun.balaji@gmail.com
