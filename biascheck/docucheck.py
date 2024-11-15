import spacy
from .utils import load_terms

class DocuCheck:
    def __init__(self, data, document=None, terms=None):
        """
        Parameters:
            data (str): Text data.
            document (str): Path to the document (optional).
            terms (str or list): Path to terms file or list of terms.
        """
        self.data = self._load_data(data, document)
        self.terms = load_terms(terms)
        self.nlp = spacy.load("en_core_web_sm")

    def _load_data(self, data, document):
        if document:
            with open(document, "r", encoding="utf-8") as file:
                return file.read()
        return data

    def analyze(self):
        """
        Analyze the document for bias based on terms.
        Returns:
            dict: Bias scores and flagged sentences.
        """
        doc = self.nlp(self.data)
        flagged_sentences = [
            sent.text for sent in doc.sents if any(term in sent.text for term in self.terms)
        ]

        bias_score = len(flagged_sentences) / max(len(list(doc.sents)), 1)
        return {
            "bias_score": bias_score,
            "flagged_sentences": flagged_sentences,
        }