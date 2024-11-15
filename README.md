# biascheck

An open-source library for checking and analyzing bias in language models, documents, datasets, and databases. The library is designed to cater to researchers, developers, and practitioners by offering a streamlined API, CLI tools, and extensibility for custom pipelines.

## Functionalities:
1) LLM Integration: Easily initialize and evaluate LLMs with minimal configuration.
Example:'''python
from biascheck import moducheck
result = moducheck(data, model=llm, israg=True)
'''
2) Automated RAG Pipeline:
Automatically build a simple Retrieval-Augmented Generation (RAG) pipeline when analyzing documents.

3) Context-Aware Bias Detection:
Upload custom lists of polarizing terms to adapt the library for local contexts.
	4.	CLI and Pip Support:
Run bias checks via command-line tools or use it as a Python library.
	5.	Sample Notebooks:
Tutorials and examples for common use cases.
	6.	Future-Proof for Multimodal Inputs:
Designed to extend support for image, video, and audio inputs in later versions.
# Classes
1) docucheck - this will measure bias/polarisation in a pdf/text etc. They can enter an optional text file if they want with key words such as local terms and their meaning (in the format of word:meaning), this allows for local language terms to be identified to find bias in every context
```python
biascheck.docucheck(data:Any, document:Optional, terms:Optional)
```

2) moducheck - this will measure bias/polarisation in a model etc. If a person enters a document, it will turn into a simple RAG model, they can further customise it by choosing their own pipeline/retreivers. They can enter an optional text file if they want with key words such as local terms and their meaning (in the format of word:meaning), this allows for local language terms to be identified to find bias in every context
```python
biascheck.moducheck(data:Any, model:Any, document:Optional, terms:Optional, retreiver:Optional, israg=False)
```

3) setcheck - this will measure bias/polarisation in a model etc. If a person enters a dataset, they can check if their dataset is skewed or biased. They can enter an optional text file if they want with key words such as local terms and their meaning (in the format of word:meaning), this allows for local language terms to be identified to find bias in every context
```python
biascheck.setcheck(data:Any, inputCols=[], terms:Optional)
```

4) basecheck - this will measure the bias/polarisation in your whole database, also you can generate a report with the most frequently occuring terms. Both graph and vector databases will work.
```python
biascheck.basecheck(data:Any, inputCols=[], terms:Optional)
```
