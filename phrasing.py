
from sklearn.feature_extraction.text import CountVectorizer

def extract_phrases(text_column, ngram_range=(1, 2), min_df=1):
    """
    Extracts phrases (n-grams) from a text column.

    Parameters:
    - text_column (iterable): A column or list of text data.
    - ngram_range (tuple): The range of n-grams to extract (default is (1, 2)).
    - min_df (int): Minimum document frequency for a phrase to be included.

    Returns:
    - list: Extracted phrases as a list of strings.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    X = vectorizer.fit_transform(text_column)
    phrases = vectorizer.get_feature_names_out()
    return phrases
