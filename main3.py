from nltk.corpus import stopwords
import re
import spacy


nlp = spacy.load("ru_core_news_sm")


# Токенизация текста
def tokenize(text):
    text = text.lower()
    tokens = text.split()
    return tokens


# Чистка текста
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('russian'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


# Лемматизация текста
def lemmatize_text(cleaned_text):
    doc = nlp(cleaned_text)
    return " ".join([token.lemma_ for token in doc])


# Предварительная подготовка текста
def process_text(text):
    tokens = tokenize(original_text)
    # print(tokens)
    tokenize_text = ', '.join(tokens)
    cleaned_text = clean_text(tokenize_text)
    # print(cleaned_text)
    lemmatized_text = lemmatize_text(cleaned_text)
    # print(lemmatized_text)
    return lemmatized_text


original_text = "Синее небо над головой. Кошка прыгнула на стол."
print(process_text(original_text))
