import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from py2neo import Graph
from langchain.vectorstores import FAISS
from biascheck.utils.embed_utils import Embedder
from biascheck.utils.terms_loader import load_terms


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
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
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
        result = self.sentiment_analyzer(text)
        sentiment = result[0]["label"]
        score = result[0]["score"]
        return sentiment, score

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

        results = []
        for hypothesis in hypotheses:
            inputs = self.contextual_tokenizer(
                text, hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=512
            )
            outputs = self.contextual_model(**inputs)
            score = outputs.logits.softmax(dim=1).tolist()[0]
            results.append(score)

        scores = {hyp: result[2] for hyp, result in zip(hypotheses, results)}
        final_classification = max(scores, key=scores.get)

        return {"scores": scores, "classification": final_classification}

    def _chunk_text(self, text, max_length=512):
        """
        Chunk long text into smaller segments.

        Parameters:
            text (str): The text to chunk.
            max_length (int): Maximum chunk size.

        Returns:
            list: List of text chunks.
        """
        words = text.split()
        return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

    def _analyze_vectordb(self):
        """
        Analyze a vector database (FAISS) for bias, sentiment, and contextual analysis.

        Returns:
            pd.DataFrame: Results of the analysis.
        """
        all_results = []

        for doc in self.data.similarity_search(query=" ", k=len(self.data.index)):
            text = doc.page_content
            metadata = doc.metadata

            # Chunk text if necessary
            text_chunks = self._chunk_text(text)

            for chunk in text_chunks:
                chunk_embedding = self.embedder.embed([chunk])[0]
                term_embeddings = self.embedder.embed(self.terms)
                similarity = cosine_similarity([chunk_embedding], term_embeddings).mean()

                sentiment, sentiment_score = self._sentiment_analysis(chunk)
                context_result = self._contextual_analysis(chunk)

                all_results.append({
                    "text": chunk,
                    "metadata": metadata,
                    "similarity": similarity,
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                    **context_result,
                })

        return pd.DataFrame(all_results)

    def _analyze_graphdb(self):
        """
        Analyze a graph database (Neo4j) for bias, sentiment, and contextual analysis.

        Returns:
            pd.DataFrame: Results of the analysis.
        """
        query = "MATCH (n) RETURN n"
        records = self.data.run(query)

        all_results = []

        for record in records:
            properties = record["n"]
            for key, value in properties.items():
                if not isinstance(value, str):
                    continue

                # Chunk text if necessary
                text_chunks = self._chunk_text(value)

                for chunk in text_chunks:
                    chunk_embedding = self.embedder.embed([chunk])[0]
                    term_embeddings = self.embedder.embed(self.terms)
                    similarity = cosine_similarity([chunk_embedding], term_embeddings).mean()

                    sentiment, sentiment_score = self._sentiment_analysis(chunk)
                    context_result = self._contextual_analysis(chunk)

                    all_results.append({
                        "text": chunk,
                        "key": key,
                        "similarity": similarity,
                        "sentiment": sentiment,
                        "sentiment_score": sentiment_score,
                        **context_result,
                    })

        return pd.DataFrame(all_results)

    def analyze(self):
        """
        Analyze the database for bias, sentiment, and contextual analysis.

        Returns:
            pd.DataFrame: Results of the analysis.
        """
        if isinstance(self.data, FAISS):
            return self._analyze_vectordb()
        elif isinstance(self.data, Graph):
            return self._analyze_graphdb()
        else:
            raise ValueError("Unsupported database type. Use FAISS or Neo4j Graph.")

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