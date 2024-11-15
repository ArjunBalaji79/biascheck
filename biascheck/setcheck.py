import pandas as pd
from .utils import load_terms

class SetCheck:
    def __init__(self, data, inputCols=[], terms=None):
        """
        Parameters:
            data (pd.DataFrame): Dataset to analyze.
            inputCols (list): Columns to analyze for bias.
            terms (str or list): Path to terms file or list of terms.
        """
        self.data = data
        self.inputCols = inputCols
        self.terms = load_terms(terms)

    def analyze(self):
        """
        Analyze the dataset for bias.
        Returns:
            dict: Bias metrics for each column.
        """
        metrics = {}
        for col in self.inputCols:
            column_data = self.data[col].astype(str)
            flagged_rows = column_data[column_data.apply(lambda x: any(term in x for term in self.terms))]
            metrics[col] = {
                "bias_score": len(flagged_rows) / max(len(column_data), 1),
                "flagged_rows": flagged_rows.tolist(),
            }
        return metrics