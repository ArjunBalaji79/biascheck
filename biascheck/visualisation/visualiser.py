import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Visualiser:
    @staticmethod
    def plot_flagged_sentences(flagged_sentences, title="Top Flagged Sentences by Bias Score"):
        """
        Plot a horizontal bar chart for flagged sentences based on bias scores.

        Parameters:
            flagged_sentences (list): List of dictionaries with keys `sentence` and `similarity`.
            title (str): Title of the plot.
        """
        sentences = [item["sentence"][:50] + "..." for item in flagged_sentences]
        scores = [item["similarity"] for item in flagged_sentences]

        plt.figure(figsize=(10, 6))
        plt.barh(sentences, scores, color="skyblue")
        plt.xlabel("Bias Score")
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    @staticmethod
    def plot_flagged_vs_non_flagged(flagged_count, total_count, title="Flagged vs Non-Flagged Sentences"):
        """
        Plot a pie chart for flagged vs. non-flagged sentences.

        Parameters:
            flagged_count (int): Number of flagged sentences.
            total_count (int): Total number of sentences.
            title (str): Title of the plot.
        """
        labels = ["Flagged", "Not Flagged"]
        sizes = [flagged_count, total_count - flagged_count]
        colors = ["red", "green"]

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_bias_vs_sentiment(dataframe, title="Bias Score vs Sentiment"):
        """
        Plot a scatter plot for bias scores vs sentiment scores.

        Parameters:
            dataframe (pd.DataFrame): DataFrame containing `bias_score` and `sentiment` columns.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(dataframe["bias_score"], dataframe["sentiment"], color="blue", s=100)
        plt.xlabel("Bias Score")
        plt.ylabel("Sentiment Score")
        plt.title(title)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_flagged_bias_scores(dataframe, title="Flagged Bias Scores"):
        """
        Plot a bar chart for bias scores of flagged records.

        Parameters:
            dataframe (pd.DataFrame): DataFrame containing `bias_score` and flagged text.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(dataframe["id"], dataframe["bias_score"], color="purple")
        plt.xlabel("Record ID")
        plt.ylabel("Bias Score")
        plt.title(title)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_bias_categories(dataframe, title="Bias Categories in Flagged Records"):
        """
        Plot a grouped bar chart for bias categories.

        Parameters:
            dataframe (pd.DataFrame): DataFrame with bias category columns (`stereotypical`, `cultural`, etc.).
            title (str): Title of the plot.
        """
        categories = ["stereotypical", "cultural", "representation", "contextual"]
        category_counts = [dataframe[cat].sum() for cat in categories]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=categories, y=category_counts, palette="viridis")
        plt.ylabel("Number of Flagged Records")
        plt.title(title)
        plt.grid(axis="y")
        plt.show()