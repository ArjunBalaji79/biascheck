import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer


class SetCheck:
    def __init__(
        self,
        data,
        input_cols=None,
        terms=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_contextual_analysis=False,
        similarity_threshold=0.5,
        verbose=False,
    ):
        """
        Analyze a dataset for bias, fairness, and polarization.

        Parameters:
            data (list or DataFrame): Dataset to analyze (list of dictionaries or Pandas DataFrame).
            input_cols (list): List of columns containing text to analyze.
            terms (str or list): Terms to check for bias (optional).
            model_name (str): Transformer model for embedding.
            use_contextual_analysis (bool): Use contextual reasoning for bias detection.
            similarity_threshold (float): Threshold for similarity to flag a sentence as biased.
            verbose (bool): Whether to print intermediate results for debugging.
        """
        self.data = data
        self.input_cols = input_cols or []
        self.terms = terms
        self.model_name = model_name
        self.verbose = verbose
        self.use_contextual_analysis = use_contextual_analysis
        self.similarity_threshold = similarity_threshold

        # Load embedding model
        self.embedder = SentenceTransformer(model_name)

        # Embed user-provided terms
        self.term_embeddings = None
        if self.terms:
            self.term_embeddings = self.embedder.encode(self.terms, convert_to_tensor=True)

        # Load contextual analysis model if enabled
        if self.use_contextual_analysis:
            self.contextual_model_name = "facebook/bart-large-mnli"
            self.contextual_tokenizer = AutoTokenizer.from_pretrained(self.contextual_model_name)
            self.contextual_model = AutoModelForSequenceClassification.from_pretrained(self.contextual_model_name)

        # Load sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

    def _sentiment_analysis(self, sentence):
        """
        Perform advanced sentiment analysis on a sentence.
        """
        try:
            result = self.sentiment_analyzer(sentence)
            sentiment = result[0]["label"]
            score = result[0]["score"]
        except Exception:
            # Fallback to TextBlob if sentiment analysis fails
            blob = TextBlob(sentence)
            sentiment = "negative" if blob.sentiment.polarity < 0 else "positive"
            score = blob.sentiment.polarity
        return sentiment, score

    def _contextual_analysis(self, sentence):
        """
        Perform contextual bias analysis using natural language inference.
        """
        hypotheses = [
            "This sentence promotes discrimination.",
            "This sentence is fair and unbiased.",
            "This sentence is offensive.",
        ]

        try:
            inputs = [
                self.contextual_tokenizer(sentence, hypothesis, return_tensors="pt", truncation=True)
                for hypothesis in hypotheses
            ]
            outputs = [self.contextual_model(**input_) for input_ in inputs]
            predictions = [output.logits.softmax(dim=-1).detach().numpy().squeeze() for output in outputs]
            return {hypotheses[i]: predictions[i] for i in range(len(hypotheses))}
        except Exception as e:
            if self.verbose:
                print(f"Contextual analysis failed for sentence: {sentence}. Error: {e}")
            return {}

    def _process_text(self, text):
        """
        Analyze text for bias and polarization.
        """
        sentences = text.split(".")
        processed = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Perform sentiment analysis
            sentiment, sentiment_score = self._sentiment_analysis(sentence)

            # Perform contextual analysis
            context_result = None
            if self.use_contextual_analysis:
                context_result = self._contextual_analysis(sentence)

            # Compute similarity with user-provided terms
            similarity_score = 0
            if self.term_embeddings is not None:
                sentence_embedding = self.embedder.encode([sentence], convert_to_tensor=True)
                similarity_score = cosine_similarity(
                    sentence_embedding.cpu().numpy(),
                    self.term_embeddings.cpu().numpy(),
                ).max()

            # Determine if the sentence is flagged
            flagged = similarity_score > self.similarity_threshold

            processed.append(
                {
                    "sentence": sentence,
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                    "similarity": similarity_score,
                    "flagged": flagged,
                    "contextual_analysis": context_result,
                }
            )

        return processed

    def analyze(self):
        """
        Analyze the dataset for bias, fairness, and polarization.

        Returns:
            DataFrame: A DataFrame containing structured results with the specified columns.
        """
        if isinstance(self.data, pd.DataFrame):
            data_type = "DataFrame"
        elif isinstance(self.data, list):
            data_type = "List"
        else:
            raise ValueError("`data` must be a list of dictionaries or a Pandas DataFrame.")

        if not self.input_cols:
            raise ValueError("`input_cols` must be provided to specify text fields for analysis.")

        structured_results = []
        for idx, record in tqdm(self.data.iterrows() if data_type == "DataFrame" else enumerate(self.data), desc="Analyzing Records"):
            for col in self.input_cols:
                text = record[col] if data_type == "List" else record[col]
                processed = self._process_text(text)
                for result in processed:
                    hypotheses = result.get("contextual_analysis", {})
                    hypothesis_columns = {}
                    
                    for key, value in hypotheses.items():
                        if isinstance(value, np.ndarray) and len(value) > 1:
                            hypothesis_columns[key] = value[1]  # Take the entailment score
                        else:
                            hypothesis_columns[key] = None  # Default to None if malformed

                    final_hypothesis = None
                    if (
                        "This sentence promotes discrimination." in hypotheses
                        and hypotheses["This sentence promotes discrimination."][1] > 0.5
                    ):
                        final_hypothesis = "Discriminatory"
                    elif (
                        "This sentence is fair and unbiased." in hypotheses
                        and hypotheses["This sentence is fair and unbiased."][1] > 0.5
                    ):
                        final_hypothesis = "Fair"
                    else:
                        final_hypothesis = "Neutral/Unclear"

                    structured_results.append({
                        "sentence": result["sentence"],
                        "sentiment": result["sentiment"],
                        "sentiment_score": result["sentiment_score"],
                        "similarity": result["similarity"],
                        "flagged": result["flagged"],
                        **hypothesis_columns,
                        "final_contextual_analysis": final_hypothesis,
                    })

        self.results_df = pd.DataFrame(structured_results)

        if self.results_df.empty:
            print("No records found.")
            return self.results_df

        return self.results_df

    def filter_dataframe(self, filter_conditions):
        """
        Filter the results DataFrame based on conditions.

        Parameters:
            filter_conditions (dict): Conditions to filter the DataFrame.

        Returns:
            DataFrame: Filtered DataFrame.
        """
        if not hasattr(self, "results_df") or self.results_df.empty:
            raise ValueError("Results DataFrame is empty. Run analyze() first.")

        filtered_df = self.results_df
        for column, condition in filter_conditions.items():
            if column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column] == condition]
            else:
                print(f"Column {column} not found in DataFrame.")

        return filtered_df
