from ..utils.embed_utils import Embedder
from ..utils.terms_loader import load_terms, preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import spacy


class DocuCheck:
    DEFAULT_TERMS = [
        "lazy", "biased", "superior", "inferior", "hate", "discrimination", "polarization"
    ]

    def __init__(
        self,
        data=None,
        document=None,
        terms=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        bias_threshold=0.4,
        use_ner=False,
        verbose=False,
    ):
        """
        Enhanced Document bias analysis class.

        Parameters:
            data (str): Raw text data to analyze (optional).
            document (str): Path to a PDF or text file (optional).
            terms (str or list): Terms to check for bias (optional).
            model_name (str): Transformer model for embedding.
            bias_threshold (float): Threshold for cosine similarity to detect bias.
            use_ner (bool): Whether to use Named Entity Recognition (NER).
            verbose (bool): Whether to print intermediate results for debugging.
        """
        self.data = self._load_data(data, document)
        self.terms = load_terms(terms) if terms else self.DEFAULT_TERMS
        self.embedder = Embedder(model_name=model_name)
        self.bias_threshold = bias_threshold
        self.use_ner = use_ner
        self.verbose = verbose
        self.nlp = spacy.load("en_core_web_sm") if use_ner else None

    def _load_data(self, data, document):
        if document:
            text = preprocess_text(document)
            return text
        if data:
            return data
        raise ValueError("Either `data` or `document` must be provided.")

    def _ner_analysis(self, text):
        """
        Perform Named Entity Recognition (NER) analysis.
        """
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def _sentiment_analysis(self, sentence):
        """
        Perform sentiment analysis on a sentence.
        """
        analysis = TextBlob(sentence)
        polarity = analysis.sentiment.polarity  # Range: [-1.0, 1.0]
        return polarity

    def analyze(self, top_n=5):
        """
        Analyze the document for bias, sentiment, and polarization.

        Parameters:
            top_n (int): Number of top biased sentences to return in verbose=False mode.

        Returns:
            dict: Results with either concise or detailed information.
        """
        if self.verbose:
            print(f"Analyzing document with {len(self.data.splitlines())} lines...")

        doc = self.nlp(self.data) if self.use_ner else None
        sentences = [sent.text.strip() for sent in doc.sents] if doc else self.data.split(".")
        term_embeddings = self.embedder.embed(self.terms)
        sentence_embeddings = self.embedder.embed(sentences)

        flagged_sentences = []
        flagged_entities = []
        sentiment_scores = []
        detailed_results = []

        for i, sentence in enumerate(sentences):
            similarity = cosine_similarity([sentence_embeddings[i]], term_embeddings).mean()
            sentiment = self._sentiment_analysis(sentence)

            if similarity > self.bias_threshold or sentiment < -0.5:  # Adjust thresholds as needed
                flagged_sentences.append(sentence)
                if self.use_ner:
                    flagged_entities.extend(self._ner_analysis(sentence))
            
            sentiment_scores.append(sentiment)
            detailed_results.append((sentence, similarity, sentiment))

            if self.verbose:
                print(f"Sentence: {sentence}\nSimilarity: {similarity:.2f}\nSentiment: {sentiment:.2f}")

        # Sort flagged sentences by similarity or sentiment
        detailed_results.sort(key=lambda x: max(x[1], abs(x[2])), reverse=True)
        top_flagged = detailed_results[:top_n]

        bias_score = len(flagged_sentences) / max(len(sentences), 1)
        overall_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        if self.verbose:
            return {
                "bias_score": bias_score,
                "overall_sentiment": overall_sentiment,
                "flagged_sentences": flagged_sentences,
                "flagged_entities": flagged_entities if self.use_ner else None,
                "sentence_sentiment_scores": sentiment_scores,
            }
        else:
            return {
                "bias_score": bias_score,
                "overall_sentiment": overall_sentiment,
                "top_flagged_sentences": [
                    {"sentence": res[0], "similarity": res[1], "sentiment": res[2]} for res in top_flagged
                ],
                "flagged_entities": flagged_entities if self.use_ner else None,
            }