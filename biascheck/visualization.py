import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_bias_scores(scores, labels):
    """
    Plot bias scores as a bar chart.
    Parameters:
        scores (list): List of bias scores.
        labels (list): Labels for each score.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(labels, scores, color="#4682B4")
    plt.title("Bias Scores")
    plt.xlabel("Entities")
    plt.ylabel("Bias Score")
    plt.show()

def generate_wordcloud(term_frequency):
    """
    Generate a word cloud from term frequencies.
    Parameters:
        term_frequency (dict): Term frequencies.
    """
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(term_frequency)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()