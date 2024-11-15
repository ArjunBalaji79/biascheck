import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_bias_scores(metrics):
    """
    Plot bias scores as a bar chart.
    Parameters:
        metrics (dict): Bias metrics with column names and scores.
    """
    columns = list(metrics.keys())
    scores = [metrics[col]["bias_score"] for col in columns]

    plt.figure(figsize=(10, 6))
    plt.bar(columns, scores, color="#4682B4")
    plt.title("Bias Scores by Column")
    plt.xlabel("Columns")
    plt.ylabel("Bias Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_wordcloud(term_frequency):
    """
    Generate a word cloud from term frequencies.
    Parameters:
        term_frequency (dict): Term frequencies for visualization.
    """
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(term_frequency)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()