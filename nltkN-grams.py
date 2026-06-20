from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import nltk

# Download required datasets
nltk.download('punkt')
nltk.download('punkt_tab')

text = "Natural Language Processing is used in Artificial Intelligence"

# Tokenization
words = word_tokenize(text.lower())

print("Tokens:")
print(words)

# Create Bigrams
bigrams = list(ngrams(words, 2))

print("\nBigrams:")
print(bigrams)

# Frequency Count
freq = Counter(bigrams)

print("\nBigram Frequency:")
for gram, count in freq.items():
    print(gram, ":", count)