from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Download required datasets
nltk.download('punkt')
nltk.download('punkt_tab')

# Hindi sentence written in English letters
text = "Bharat ek sundar desh hai aur log mehnati hain."

# Tokenization
words = word_tokenize(text)

print("Tokens:")
print(words)

# Stemming
stemmer = PorterStemmer()

stemmed_words = [stemmer.stem(word) for word in words]

print("\nStemmed Words:")
print(stemmed_words)