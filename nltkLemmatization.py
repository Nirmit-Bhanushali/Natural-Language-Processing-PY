from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

text = "The boys are playing football and running in the ground."

# Tokenization
words = word_tokenize(text)

print("Tokens:")
print(words)

# Lemmatization
lemmatizer = WordNetLemmatizer()

lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

print("\nLemmatized Words:")
print(lemmatized_words)