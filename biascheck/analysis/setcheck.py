import pandas as pd
from ..utils.embed_utils import Embedder
from ..utils.terms_loader import load_terms

class SetCheck:
    def __init__(self, data, inputCols=[], terms=None, model_name=None):
        """
        Dataset bias analysis class.
        Parameters:
            data (pd.DataFrame): Dataset to analyze.
            inputCols (list): Columns to analyze for bias.
            terms (str or list): Terms for bias detection.
            model_name (str): Transformer model for embedding.
        """
        self.data = data
        self.inputCols = inputCols
        self.terms = load_terms(terms)
        self.embedder = Embedder(model_name=model_name)

    def analyze(self):
        """
        Analyze the dataset for bias.
        Returns:
            dict: Bias metrics for each column.
        """
        metrics = {}
        for col in self.inputCols:
            column_data = self.data[col].astype(str).tolist()
            term_embeddings = self.embedder.embed(self.terms)
            column_embeddings = self.embedder.embed(column_data)

            flagged_rows = []
            for i, row in enumerate(column_data):
                similarity = cosine_similarity([column_embeddings[i]], term_embeddings).mean()
                if similarity > 0.5:  # Threshold for bias detection
                    flagged_rows.append(row)

            metrics[col] = {
                "bias_score": len(flagged_rows) / max(len(column_data), 1),
                "flagged_rows": flagged_rows,
            }
        return metrics