import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
import json
import numpy as np # Added for handling potential numpy types in DataFrame

class ReportGenerator:
    def __init__(self, analysis_df: pd.DataFrame):
        """
        Initializes the ReportGenerator with the analysis DataFrame.

        Parameters:
            analysis_df (pd.DataFrame): The DataFrame output from one of the BiasCheck analyzers.
        """
        if not isinstance(analysis_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.df = analysis_df.copy() # Use a copy to avoid modifying the original DataFrame
        self.summary = self._generate_summary_stats()

    def _generate_summary_stats(self):
        """
        Generates basic summary statistics from the DataFrame.
        Handles potential variations in column names from different analyzers.
        """
        stats = {"total_records": len(self.df)}

        if not self.df.empty:
            if "flagged" in self.df.columns:
                stats["flagged_records"] = int(self.df["flagged"].sum())
                stats["percentage_flagged"] = (stats["flagged_records"] / stats["total_records"]) * 100 if stats["total_records"] > 0 else 0

            # Handle sentiment column variations
            sentiment_col_name = None
            if "sentiment" in self.df.columns:
                sentiment_col_name = "sentiment"
            elif "sentiment_label" in self.df.columns: # e.g., ModuCheck
                sentiment_col_name = "sentiment_label"
            
            if sentiment_col_name:
                stats["sentiment_distribution"] = self.df[sentiment_col_name].value_counts().to_dict()

            # Handle contextual hypothesis column variations
            hypothesis_col_names = [
                "final_contextual_hypothesis", # ModuCheck, RAGCheck
                "final_hypothesis",           # DocuCheck
                "classification",             # BaseCheck (potentially nested)
                "final_contextual_analysis"   # SetCheck
            ]
            final_hypothesis_col = next((col for col in hypothesis_col_names if col in self.df.columns), None)

            if final_hypothesis_col:
                stats["contextual_hypothesis_distribution"] = self.df[final_hypothesis_col].value_counts().to_dict()
            elif "scores" in self.df.columns and not self.df["scores"].empty and isinstance(self.df["scores"].iloc[0], dict) and "classification" in self.df["scores"].iloc[0]:
                # Handle BaseCheck's nested structure if 'classification' is directly available after unpacking
                try:
                    classifications = self.df["scores"].apply(lambda x: x.get("classification") if isinstance(x, dict) else None)
                    stats["contextual_hypothesis_distribution"] = classifications.value_counts().to_dict()
                except Exception: # Fallback if apply fails
                    stats["contextual_hypothesis_distribution"] = "Error processing nested scores"


            # Handle bias/similarity score column variations
            bias_score_col_name = None
            if "bias_score" in self.df.columns: # ModuCheck
                bias_score_col_name = "bias_score"
            elif "similarity" in self.df.columns: # DocuCheck, SetCheck, BaseCheck
                bias_score_col_name = "similarity"
            
            if bias_score_col_name and pd.api.types.is_numeric_dtype(self.df[bias_score_col_name]):
                stats[f"average_{bias_score_col_name}"] = self.df[bias_score_col_name].mean()
            
        return stats

    def generate_text_report(self, report_path: str = "bias_analysis_report.txt"):
        """
        Generates a human-readable text report and saves it to a file.

        Parameters:
            report_path (str): Path to save the text report.
        """
        report_content = "BiasCheck Analysis Report\n"
        report_content += "=========================\n\n"
        report_content += "Summary Statistics:\n"
        report_content += "-------------------\n"
        for key, value in self.summary.items():
            if isinstance(value, dict):
                report_content += f"{key.replace('_', ' ').title()}:\n"
                for sub_key, sub_value in value.items():
                    report_content += f"  - {sub_key}: {sub_value}\n"
            elif isinstance(value, float):
                report_content += f"{key.replace('_', ' ').title()}: {value:.2f}\n"
            else:
                report_content += f"{key.replace('_', ' ').title()}: {value}\n"
        report_content += "\n"

        report_content += "Detailed Records (Sample - First 10 or fewer if less data):\n"
        report_content += "----------------------------------------------------------\n"
        
        text_col_candidates = ["text", "sentence", "response"] # Common text column names
        text_col = next((col for col in text_col_candidates if col in self.df.columns), None)

        sample_df = self.df.head(10)

        if text_col and not sample_df.empty:
            for index, row in sample_df.iterrows():
                report_content += f"\nRecord {index + 1}:\n"
                report_content += f"  Text: {row.get(text_col, 'N/A')}\n"
                
                sent_label = row.get("sentiment", row.get("sentiment_label", "N/A"))
                sent_score = row.get("sentiment_score", "N/A")
                if sent_label != "N/A":
                    report_content += f"  Sentiment: {sent_label} (Score: {sent_score if isinstance(sent_score, str) else f'{sent_score:.2f}'})\n"

                bias_val = row.get("bias_score", row.get("similarity", "N/A"))
                if bias_val != "N/A":
                     report_content += f"  Bias/Similarity Score: {bias_val if isinstance(bias_val, str) else f'{bias_val:.2f}'}\n"
                
                final_hyp_val = row.get("final_contextual_hypothesis", 
                                    row.get("final_hypothesis", 
                                    row.get("final_contextual_analysis", "N/A")))
                if final_hyp_val == "N/A" and "scores" in row and isinstance(row["scores"], dict) and "classification" in row["scores"]:
                    final_hyp_val = row["scores"]["classification"]
                
                if final_hyp_val != "N/A":
                    report_content += f"  Final Contextual Hypothesis: {final_hyp_val}\n"

                if "flagged" in row and pd.notna(row["flagged"]):
                     report_content += f"  Flagged: {row['flagged']}\n"
        elif not sample_df.empty:
             report_content += "No primary text column (e.g., text, sentence, response) found for detailed sample records.\n"
             report_content += "Showing raw sample data instead:\n"
             report_content += sample_df.to_string() + "\n"
        else:
            report_content += "No data available to display in detailed records.\n"


        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Text report saved to {report_path}")
        return report_path


    def _default_converter(self, o):
        """Helper to convert non-serializable types to string for JSON."""
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        # Add other types as needed, e.g., pd.Timestamp
        # elif isinstance(o, pd.Timestamp):
        #     return o.isoformat()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    def generate_json_report(self, report_path: str = "bias_analysis_report.json"):
        """
        Generates a JSON report containing summary statistics and the full DataFrame.
        Handles potential non-serializable numpy types.

        Parameters:
            report_path (str): Path to save the JSON report.
        """
        # Create a deep copy for serialization to avoid altering self.df
        df_for_json = self.df.copy()

        # Convert known problematic types to native Python types or strings
        for col in df_for_json.columns:
            # Handle numpy numeric types
            if df_for_json[col].dtype in [np.int64, np.int32, np.float64, np.float32]:
                 df_for_json[col] = df_for_json[col].apply(lambda x: x.item() if pd.notna(x) else None)
            # Handle numpy boolean types
            elif df_for_json[col].dtype == np.bool_:
                 df_for_json[col] = df_for_json[col].apply(lambda x: bool(x) if pd.notna(x) else None)
            # Convert object columns that might contain complex types (like dicts of numpy arrays)
            elif df_for_json[col].dtype == 'object':
                # This is a more complex case; if dicts contain numpy arrays,
                # they might need custom handling or rely on the default converter.
                # For now, we assume simple dicts or let default handle it.
                pass


        report_data = {
            "summary_statistics": self.summary,
            # Convert DataFrame to list of dicts, which is JSON friendly
            "detailed_data": df_for_json.to_dict(orient="records")
        }
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=4, default=self._default_converter)
            print(f"JSON report saved to {report_path}")
        except TypeError as e:
            print(f"Error serializing to JSON: {e}")
            print("Attempting to save with problematic fields converted to string.")
            # Fallback: convert entire DataFrame to string representation if specific conversion fails
            report_data_fallback = {
                "summary_statistics": self.summary,
                "detailed_data": self.df.astype(str).to_dict(orient="records")
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data_fallback, f, indent=4)
            print(f"JSON report saved with string conversion to {report_path}")
            
        return report_path 