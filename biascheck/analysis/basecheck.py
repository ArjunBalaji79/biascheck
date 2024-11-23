import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain.vectorstores import FAISS
from biascheck.utils.embed_utils import Embedder
from biascheck.utils.terms_loader import load_terms
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class BaseCheck:
    def __init__(
        self,
        data,
        input_cols=None,
        terms=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_contextual_analysis=False,
        use_sentiment_analysis=True,
        verbose=False,
    ):
        """
        Database bias analysis class for vector and graph databases.

        Parameters:
            data (Any): Database connection or raw data (Vector database or Graph database).
            input_cols (list): Columns or keys to analyze for bias.
            terms (str or list): Terms for bias detection.
            model_name (str): Transformer model for embedding.
            use_contextual_analysis (bool): Whether to use contextual analysis for detecting bias.
            use_sentiment_analysis (bool): Whether to perform sentiment analysis.
            verbose (bool): Whether to print intermediate results for debugging.
        """
        self.data = data
        self.input_cols = input_cols or []
        self.terms = load_terms(terms)
        self.model_name = model_name
        self.embedder = Embedder(model_name=model_name)
        self.use_contextual_analysis = use_contextual_analysis
        self.use_sentiment_analysis = use_sentiment_analysis
        self.verbose = verbose

        # Load sentiment and contextual analysis models if enabled
        if use_contextual_analysis or use_sentiment_analysis:
            self.contextual_model_name = "facebook/bart-large-mnli"
            self.contextual_tokenizer = AutoTokenizer.from_pretrained(self.contextual_model_name)
            self.contextual_model = AutoModelForSequenceClassification.from_pretrained(self.contextual_model_name)

            self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

    def _sentiment_analysis(self, text):
        """
        Perform sentiment analysis on a given text.

        Parameters:
            text (str): The text to analyze.

        Returns:
            tuple: Sentiment label and score.
        """
        try:
            result = self.sentiment_analyzer(text[:512])  # Truncate text to prevent token overflow
            sentiment = result[0]["label"]
            score = result[0]["score"]
            return sentiment, score
        except Exception as e:
            if self.verbose:
                print(f"Sentiment analysis failed: {e}")
            return "neutral", 0.0

    def _contextual_analysis(self, text):
        """
        Perform contextual analysis using a transformer-based model.

        Parameters:
            text (str): The text to analyze.

        Returns:
            dict: Scores for different hypotheses and the final classification.
        """
        hypotheses = [
            "This sentence promotes discrimination.",
            "This sentence is fair and unbiased.",
            "This sentence is offensive.",
        ]

        try:
            inputs = [
                self.contextual_tokenizer(
                    text[:512],  # Truncate text to prevent token overflow
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                )
                for hypothesis in hypotheses
            ]
            outputs = [self.contextual_model(**input_) for input_ in inputs]
            scores = {hyp: output.logits.softmax(dim=1).tolist()[0][2] for hyp, output in zip(hypotheses, outputs)}
            final_classification = max(scores, key=scores.get)
            return {"scores": scores, "classification": final_classification}
        except Exception as e:
            if self.verbose:
                print(f"Contextual analysis failed: {e}")
            return {"scores": {}, "classification": "unknown"}

    def _analyze_vectordb(self, top_k=10):
        """
        Analyze a vector database (FAISS) for bias, sentiment, and contextual analysis.

        Parameters:
            top_k (int): Number of top documents to analyze.

        Returns:
            pd.DataFrame: Results of the analysis.
        """
        all_results = []

        # Retrieve top_k documents
        docs = self.data.similarity_search(query=" ", k=min(top_k, len(self.data.docstore._dict)))
        for doc in docs:
            text = doc.page_content[:512]  # Truncate text for safe processing
            metadata = doc.metadata

            # Compute similarity with predefined terms
            chunk_embedding = self.embedder.embed([text])[0]
            term_embeddings = self.embedder.embed(self.terms)
            similarity = cosine_similarity([chunk_embedding], term_embeddings).mean()

            # Perform sentiment and contextual analysis
            sentiment, sentiment_score = self._sentiment_analysis(text)
            context_result = self._contextual_analysis(text)

            all_results.append({
                "text": text,
                "metadata": metadata,
                "similarity": similarity,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                **context_result,
            })

        return pd.DataFrame(all_results)

    def analyze(self, top_k=10):
        """
        Analyze the database for bias, sentiment, and contextual analysis.

        Parameters:
            top_k (int): Number of top documents to analyze.

        Returns:
            pd.DataFrame: Results of the analysis.
        """
        if isinstance(self.data, FAISS):
            return self._analyze_vectordb(top_k)
        else:
            raise ValueError("Unsupported database type. Use FAISS.")

    def generate_report(self, results_df):
        """
        Generate a detailed report from the analysis results.

        Parameters:
            results_df (pd.DataFrame): DataFrame of analysis results.

        Returns:
            str: A detailed report.
        """
        report = "Bias Analysis Report:\n"
        for _, row in results_df.iterrows():
            report += f"\nText: {row['text']}\n"
            report += f"Similarity: {row['similarity']:.2f}\n"
            report += f"Sentiment: {row['sentiment']} (Score: {row['sentiment_score']:.2f})\n"
            report += "Contextual Analysis Scores:\n"
            for hypothesis, score in row["scores"].items():
                report += f"  {hypothesis}: {score:.2f}\n"
            report += f"Final Classification: {row['classification']}\n"
        return report
