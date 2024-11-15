from ..utils.embed_utils import Embedder
from ..utils.terms_loader import load_terms

class DocuCheck:
    def __init__(self, data, document=None, terms=None, model_name=None):
        """
        Document bias analysis class.
        Parameters:
            data (str): Text data to analyze.
            document (str): Path to the document (optional).
            terms (str or list): Terms to check for bias.
            model_name (str): Transformer model for embedding.
        """
        self.data = self._load_data(data, document)
        self.terms = load_terms(terms)
        self.embedder = Embedder(model_name=model_name)

    def _load_data(self, data, document):
        if document:
            with open(document, "r", encoding="utf-8") as file:
                return file.read()
        return data

    def analyze(self):
        """
        Analyze the document for bias.
        Returns:
            dict: Bias score and flagged sentences.
        """
        sentences = self.data.split(".")
        term_embeddings = self.embedder.embed(self.terms)
        sentence_embeddings = self.embedder.embed(sentences)

        flagged_sentences = []
        for i, sentence in enumerate(sentences):
            similarity = cosine_similarity([sentence_embeddings[i]], term_embeddings).mean()
            if similarity > 0.5:  # Threshold for bias detection
                flagged_sentences.append(sentence)

        bias_score = len(flagged_sentences) / max(len(sentences), 1)
        return {
            "bias_score": bias_score,
            "flagged_sentences": flagged_sentences,
        }