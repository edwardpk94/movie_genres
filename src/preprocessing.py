"""
Preprocessing/cleaning functions for model inputs
"""

from typing import List

import regex as re
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Remove numberings (i.e. 1st, 2nd, 3rd, etc.) first so the letters aren't
# stranded after removing the numbers (i.e. 1st -> st, 2nd -> nd)
numbering_re = re.compile(r"(\d+(?:st|nd|rd|th))")

# Keep only alpha characters
alpha_re = re.compile(r"[^a-z]+")

# Remove multiple spaces
multi_space_re = re.compile(r"\s+")

# stop_words = set(stopwords.words('english'))
# Use an abbreviated stopword list to preserve as much contextual information as possible
stop_words = [
    "for",
    "the",
    "a",
    "an",
    "in",
    "of",
    "as",
    "at",
    "by",
    "for",
    "on",
    "and",
    "to",
]

# Don't forget to download wordnet if you haven't already!
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def dummy_function(text: str):
    """
    Dummy function for use with tf-idf vectorizer. This enables text cleaning
    prior to tokenization
    """
    return text


def clean_text(text: str) -> List[str]:
    """
    Converting to lowercase, keeping only alpha characters, and lemmatizing

    Only removing a select number of stopwords since many stopwords may
    contain useful contextual information for classification

    Returns:
        List of tokens to be fed into model
    """
    # Convert unicode to ascii version (to preserve words like Pokémon)
    text = unidecode(text)

    # Lowercase, remove numberings (i.e. 1st, 2nd)
    text = text.lower()
    text = numbering_re.sub(" ", text)

    # Remove apostrophes directly as they cause issues with the tokenizer
    text = text.replace("'", "")

    # Keep only alpha characters
    text = alpha_re.sub(" ", text)

    # Turn multiple spaces into a single space
    text = multi_space_re.sub(" ", text)

    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]

    # Try without removing stop words first since they may be important for classification
    # tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


if __name__ == "__main__":
    pass
