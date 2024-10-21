from nltk.corpus import stopwords
import re
import spacy


def tokenize(text):
    text = text.lower()
    tokens = text.split()
    return tokens


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('russian'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


original_text = "Синее небо над головой. Кошка прыгнула на стол."
tokens = tokenize(original_text)
print(tokens)
tokenize_text = ', '.join(tokens)


cleaned_text = clean_text(tokenize_text)
print(cleaned_text)


nlp = spacy.load("ru_core_news_sm")


# Функция для лемматизации текста
def lemmatize_text(cleaned_text):
    doc = nlp(cleaned_text)
    return " ".join([token.lemma_ for token in doc])


print(lemmatize_text(cleaned_text))
