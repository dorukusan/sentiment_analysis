from nltk.corpus import stopwords
from collections import Counter
from spellchecker import SpellChecker
import re
import spacy
import pandas as pd


# Загрузка TSKV файла с данными
data = pd.read_csv('geo-reviews-dataset-2023.tskv', sep='\t', header=None)

# Установка имен столбцов
data.columns = ['address', 'name_ru', 'rating', 'rubrics', 'text']

# Фильтр только столбцов с текстом отзыва и оценкой
data = data[['text', 'rating']]

# Снимаем ограничения вывода таблицы
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# Смотрим на данные, выводим 10 первых строк
print(data[:10])

nlp = spacy.load("ru_core_news_sm")
spell = SpellChecker(language='ru')


# Исправление опечаток
def correct_spelling(word):
    if spell.unknown([word]):
        corrected_word = spell.candidates(word)
        if corrected_word:
            return corrected_word.pop()
    return word


# Токенизация текста
def tokenize(text):
    text = text.lower()
    tokens = text.split()
    return tokens


# Чистка текста
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('russian'))
    filtered_words = [word for word in text.split() if word not in stop_words]
    corrected_words = [correct_spelling(word) for word in filtered_words]
    return " ".join(corrected_words)


# Лемматизация текста
def lemmatize_text(cleaned_text):
    doc = nlp(cleaned_text)
    return " ".join([token.lemma_ for token in doc])


# Предварительная обработка текста и заполнение словаря
def process_text(text, dictionary):
    tokens = tokenize(text)
    tokenize_text = ', '.join(tokens)
    # print(tokenize_text)
    cleaned_text = clean_text(tokenize_text)
    # print(cleaned_text)
    lemmatized_text = lemmatize_text(cleaned_text)
    for token in lemmatized_text.split():
        if token not in dictionary:
            dictionary.append(token)
    # print(lemmatized_text)

    return lemmatized_text


# Вектор слов
def vectorize_text(text, dictionary):
    vector = [0] * len(dictionary)
    token_counts = Counter(text.split())
    for i, word in enumerate(dictionary):
        vector[i] = token_counts.get(word, 0)
    return vector


dictionary = []

original_text1 = "Синее небо над головоц. Кошка прыгнула на стол."
original_text2 = "Синее небо над головой. Кошка прыгнула."
# process_text(original_text1, dictionary)
processed_text = process_text(original_text1, dictionary)
print(processed_text)
# print(process_text(original_text2, dictionary))
print(*dictionary)
print(*vectorize_text(processed_text, dictionary))
