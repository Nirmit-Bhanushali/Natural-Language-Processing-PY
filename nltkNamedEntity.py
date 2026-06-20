import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

# Download required datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

text = "Elon Musk founded Tesla in California."

# Tokenization
words = word_tokenize(text)

# POS Tagging
pos_tags = pos_tag(words)

print("POS Tags:")
print(pos_tags)

# Named Entity Recognition
ner = ne_chunk(pos_tags)

print("\nNamed Entities:")
print(ner)