import pandas as pd
from ..utils.embed_utils import Embedder
from ..utils.terms_loader import load_terms
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob


class SetCheck:
    def __init__(
        self,
        data,
        input_cols=None,
        terms=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        bias_threshold=0.4,
        verbose=False,
    ):
        """
        Analyze a dataset for bias, sentiment, and polarization.

        Parameters:
            data (list or DataFrame): Dataset to analyze (list of dictionaries or Pandas DataFrame).
            input_cols (list): List of columns containing text to analyze.
            terms (str or list): Terms to check for bias (optional).
            model_name (str): Transformer model for embedding.
            bias_threshold (float): Threshold for cosine similarity to detect bias.
            verbose (bool): Whether to print intermediate results for debugging.
        """
        self.data = data
        self.input_cols = input_cols or []  # Ensure input columns are specified
        self.terms = load_terms(terms) if terms else []
        self.embedder = Embedder(model_name=model_name)
        self.bias_threshold = bias_threshold
        self.verbose = verbose

    def _sentiment_analysis(self, sentence):
        """
        Perform sentiment analysis on a sentence.
        """
        analysis = TextBlob(sentence)
        return analysis.sentiment.polarity  # Range: [-1.0, 1.0]

    def _process_text(self, text):
        """
        Process and analyze text for bias.
        """
        sentences = text.split(".")
        sentence_embeddings = self.embedder.embed(sentences)
        term_embeddings = self.embedder.embed(self.terms)

        flagged = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            similarity = cosine_similarity([sentence_embeddings[i]], term_embeddings).mean()
            sentiment = self._sentiment_analysis(sentence)

            if self.verbose:
                print(f"Sentence: {sentence}\nSimilarity: {similarity:.2f}\nSentiment: {sentiment:.2f}")

            if similarity > self.bias_threshold or sentiment < -0.5:
                flagged.append(
                    {
                        "flagged_text": sentence.strip(),
                        "bias_score": similarity,
                        "sentiment": sentiment,
                    }
                )
        return flagged

    def analyze(self, top_n=5):
        """
        Analyze the dataset for bias.

        Parameters:
            top_n (int): Number of top biased records to return.

        Returns:
            DataFrame: A DataFrame containing flagged records and their analysis.
        """
        if isinstance(self.data, pd.DataFrame):
            data_type = "DataFrame"
        elif isinstance(self.data, list):
            data_type = "List"
        else:
            raise ValueError("`data` must be a list of dictionaries or a Pandas DataFrame.")

        if not self.input_cols:
            raise ValueError("`input_cols` must be provided to specify text fields for analysis.")

        flagged_records = []
        for idx, record in (self.data.iterrows() if data_type == "DataFrame" else enumerate(self.data)):
            for col in self.input_cols:
                if data_type == "List" and col not in record:
                    continue

                text = record[col] if data_type == "List" else record[col]  # Handle Series for DataFrame
                flagged = self._process_text(text)
                for flag in flagged:
                    flagged_records.append(
                        {
                            "id": record["id"] if data_type == "List" else idx,  # Extract ID correctly
                            "column": col,
                            **flag,
                        }
                    )

        # Convert flagged records to a DataFrame
        flagged_df = pd.DataFrame(flagged_records)

        if flagged_df.empty:
            print("No flagged records found.")
            return flagged_df

        # Sort flagged records by bias score or sentiment
        flagged_df = flagged_df.sort_values(
            by=["bias_score", "sentiment"], ascending=[False, True]
        ).reset_index(drop=True)

        # Limit to top_n flagged records if specified
        return flagged_df.head(top_n)