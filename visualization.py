
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

def visualize_word_clouds(df_incorrect, dataset):
    text = " ".join(df_incorrect['hypothesis'].tolist())
    wordcloud = WordCloud(stopwords=set(), background_color="white", max_words=100).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def analyze_common_words(df_incorrect, dataset):
    def find_top_common_non_stopwords(data, top_n=10):
        all_text = " ".join(data)
        words = re.findall(r'\b\w+\b', all_text.lower())
        non_stopwords = [word for word in words if word not in set()]
        word_counts = Counter(non_stopwords)
        return word_counts.most_common(top_n)

    top_words = find_top_common_non_stopwords(df_incorrect['hypothesis'].tolist())
    print("Top words:", top_words)
