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


# Предварительная обработка текста и заполнение словаря
def process_text(text, dictionary):
    tokens = tokenize(text)
    # print(tokens)
    tokenize_text = ', '.join(tokens)
    cleaned_text = clean_text(tokenize_text)
    # print(cleaned_text)
    lemmatized_text = lemmatize_text(cleaned_text)
    for token in lemmatized_text.split():
        if token not in dictionary:
            dictionary.append(token)
    # print(lemmatized_text)

    return lemmatized_text


dictionary = []

original_text1 = "Синее небо над головой. Кошка прыгнула на стол."
original_text2 = "Синее небо над головой. Кошка на стол прыгнула."
print(process_text(original_text1, dictionary))
print(process_text(original_text2, dictionary))
print(*dictionary)
