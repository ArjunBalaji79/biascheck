from ..utils.embed_utils import Embedder
from ..utils.terms_loader import load_terms, preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
import pandas as pd


class DocuCheck:
    def __init__(
        self,
        data=None,
        document=None,
        terms=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        bias_threshold=0.4,
        use_ner=False,
        use_advanced_sentiment=False,
        use_contextual_analysis=False,
        verbose=False,
    ):
        """
        Advanced Document bias analysis class with hypothesis-based contextual analysis.

        Parameters:
            data (str): Raw text data to analyze (optional).
            document (str): Path to a PDF or text file (optional).
            terms (str or list): Terms to check for bias (optional).
            model_name (str): Transformer model for embedding.
            bias_threshold (float): Threshold for cosine similarity to detect bias.
            use_ner (bool): Whether to use Named Entity Recognition (NER).
            use_advanced_sentiment (bool): Use transformer-based sentiment analysis.
            use_contextual_analysis (bool): Use hypothesis-based contextual analysis for bias reasoning.
            verbose (bool): Whether to print intermediate results for debugging.
        """
        self.data = self._load_data(data, document)
        self.terms = load_terms(terms) if terms else None
        self.embedder = Embedder(model_name=model_name) if self.terms else None
        self.bias_threshold = bias_threshold
        self.use_ner = use_ner
        self.verbose = verbose
        self.use_advanced_sentiment = use_advanced_sentiment
        self.use_contextual_analysis = use_contextual_analysis
        self.nlp = spacy.load("en_core_web_sm") if use_ner else None

        # Load advanced sentiment and contextual analysis models if enabled
        if self.use_advanced_sentiment:
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        if self.use_contextual_analysis:
            self.contextual_model_name = "facebook/bart-large-mnli"
            self.contextual_tokenizer = AutoTokenizer.from_pretrained(self.contextual_model_name)
            self.contextual_model = AutoModelForSequenceClassification.from_pretrained(self.contextual_model_name)

    def _load_data(self, data, document):
        """
        Load the data from raw text or a document file.
        """
        if document:
            return preprocess_text(document)
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
        if self.use_advanced_sentiment:
            result = self.sentiment_analyzer(sentence)
            sentiment = result[0]["label"]
            score = result[0]["score"]
            return sentiment, score
        else:
            from textblob import TextBlob
            analysis = TextBlob(sentence)
            polarity = analysis.sentiment.polarity
            return "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral", polarity

    def _contextual_analysis(self, sentence):
        """
        Perform hypothesis-based contextual analysis using a transformer-based model.
        """
        hypotheses = [
            "This sentence promotes discrimination.",
            "This sentence is fair and unbiased.",
            "This sentence is offensive.",
        ]
        inputs = [
            self.contextual_tokenizer(sentence, hypothesis, return_tensors="pt", truncation=True)
            for hypothesis in hypotheses
        ]
        outputs = [self.contextual_model(**input_) for input_ in inputs]
        predictions = [output.logits.softmax(dim=1)[0].tolist() for output in outputs]
        return {hypotheses[i]: predictions[i][2] for i in range(len(hypotheses))}  # Use entailment score

    def analyze(self):
        """
        Analyze the document for bias, sentiment, and polarization.

        Returns:
            pd.DataFrame: A DataFrame with analysis results for each sentence.
        """
        if self.verbose:
            print(f"Analyzing document with {len(self.data.splitlines())} lines...")

        doc = self.nlp(self.data) if self.use_ner else None
        sentences = [sent.text.strip() for sent in doc.sents] if doc else self.data.split(".")
        term_embeddings = self.embedder.embed(self.terms) if self.terms else None
        sentence_embeddings = self.embedder.embed(sentences) if self.terms else None

        results = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Sentiment analysis
            sentiment, sentiment_score = self._sentiment_analysis(sentence)

            # Contextual analysis
            context_result = None
            if self.use_contextual_analysis:
                context_result = self._contextual_analysis(sentence)

            # Similarity analysis
            similarity = None
            if self.terms:
                similarity = cosine_similarity([sentence_embeddings[i]], term_embeddings).mean()

            # Determine the final hypothesis from contextual analysis
            if context_result:
                final_hypothesis = max(context_result, key=context_result.get)
            else:
                final_hypothesis = "Not Analyzed"

            results.append({
                "sentence": sentence,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "similarity": similarity,
                **context_result,
                "final_hypothesis": final_hypothesis,
            })

        return pd.DataFrame(results)