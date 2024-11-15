# biascheck
An open source library for checking bias 

## Functionalities:
1) Input an LLM with ease by initialising it through an instance like langchain.llms
2) If you upload a document, it will automatically build a simple RAG pipeline (further customaisation can be added to choose which retreiver
3) You can also upload a list of polarising terms - relevant to the context you want to check in, for example it changes from country to country (helps with NER, etc..)
4) All of this will be usable from the CLI/pip library/repo
5) there will be some sample notebooks
6) Currently it will check for language inputs but eventually will be extended for multimodal input

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
