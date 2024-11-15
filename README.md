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
1) documeasure - this will enable a person to measure bias/polarisation in a pdf/text etc. They can enter an optional text file if they want with key words such as local terms and their meaning (in the format of word:meaning), this allows for local language terms to be identified to find bias in every context
```python
biascheck.documeasure(data:Any, model:Any, document:Optional, terms:Optional)
```

