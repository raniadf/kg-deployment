import re
from collections import Counter
import math
import os
import nltk
import spacy
import ssl
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Only need to download once
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2000000

class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def read_texts(self, directory_path):
        all_texts = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r') as file:
                    all_texts.append(file.read())
        self.text = " ".join(all_texts)

    def preprocess_text(self, text):
        text = text.lower()
        text = text.replace('/', ' ')
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'^#.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_filter(self, text):
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return filtered_tokens

    def lemmatize_tokens(self, tokens):
        tokens = nlp(' '.join(tokens))
        lemmatized_tokens = [token.lemma_ for token in tokens]
        return lemmatized_tokens

class InformationContentCalculator(TextProcessor):
    def __init__(self, file_path):
        super().__init__()
        self.read_texts(file_path)
        self.tokens = self.lemmatize_tokens(self.tokenize_and_filter(self.preprocess_text(self.text)))
        self.synonym_map = self.create_synonym_map(self.tokens)

    def create_synonym_map(self, tokens):
        synonym_map = {}
        for token in tokens:
            synonyms = set()
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    if lemma.name() != token:
                        synonyms.add(self.lemmatizer.lemmatize(lemma.name().replace('_', ' ')))
            synonym_map[token] = synonyms
        return synonym_map

    def calculate_ic(self, word):
        freq_count = Counter()
        for token in self.tokens:
            freq_count[token] += 1
            if token in self.synonym_map:
                for synonym in self.synonym_map[token]:
                    freq_count[synonym] += 1

        total_words = len(self.tokens) + len(self.synonym_map)
        frequency = freq_count.get(word, 0) / total_words
        print(f"Frequency of '{word}': {frequency}")
        ic = -math.log(frequency) if frequency > 0 else 0

        return ic

if __name__ == "__main__":
    directory_path = "./data/text"
    word_to_check = 'recursive'

    ic_calculator = InformationContentCalculator(directory_path)

    ic = ic_calculator.calculate_ic(word_to_check)
    print(f"Information content of '{word_to_check}': {ic}")