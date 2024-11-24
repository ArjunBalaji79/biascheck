import matplotlib.pyplot as plt
import seaborn as sns


class Visualiser:
    @staticmethod
    def plot_sentiment_distribution(dataframe, title="Sentiment Distribution"):
        """
        Plot a bar chart for sentiment distribution.

        Parameters:
            dataframe (pd.DataFrame): DataFrame containing a `sentiment` column.
            title (str): Title of the plot.
        """
        sentiment_counts = dataframe["sentiment"].value_counts()

        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_contextual_analysis_scores(dataframe, title="Contextual Analysis Scores"):
        """
        Plot a grouped bar chart for contextual analysis scores.

        Parameters:
            dataframe (pd.DataFrame): DataFrame with contextual analysis score columns.
            title (str): Title of the plot.
        """
        score_columns = [col for col in dataframe.columns if col.startswith("This sentence")]
        scores = dataframe[score_columns].mean()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=scores.index, y=scores.values, palette="mako")
        plt.xlabel("Contextual Hypotheses")
        plt.ylabel("Average Score")
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.show()

    @staticmethod
    def plot_sentences_by_sentiment_score(dataframe, title="Sentences by Sentiment Score"):
        """
        Plot a scatter plot of sentences based on sentiment scores.

        Parameters:
            dataframe (pd.DataFrame): DataFrame containing `sentiment_score` and `sentence`.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(
            dataframe["sentiment_score"], range(len(dataframe)),
            color="blue", alpha=0.7
        )
        plt.xlabel("Sentiment Score")
        plt.ylabel("Sentence Index")
        plt.title(title)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_final_contextual_analysis(dataframe, title="Final Contextual Analysis Distribution"):
        """
        Plot a pie chart for the distribution of final contextual analysis results.

        Parameters:
            dataframe (pd.DataFrame): DataFrame containing `final_contextual_analysis` column.
            title (str): Title of the plot.
        """
        analysis_counts = dataframe["final_contextual_analysis"].value_counts()

        plt.figure(figsize=(8, 8))
        plt.pie(analysis_counts.values, labels=analysis_counts.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
        plt.title(title)
        plt.show()
