from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from console_progressbar import ProgressBar
import re
import spacy
import pandas as pd


nlp = spacy.load("ru_core_news_sm")


# Токенизация текста
def tokenize(text):
    text = text.lower()
    tokens = text.split()
    return tokens


# Чистка текста
def clean_text(text):
    text = text.replace("\\n", " ")
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('russian'))
    filtered_words = [word for word in text.split() if word not in stop_words]
    return " ".join(filtered_words)


# Лемматизация текста
def lemmatize_text(cleaned_text):
    doc = nlp(cleaned_text)
    return " ".join([token.lemma_ for token in doc])


# Предварительная обработка текста и заполнение словаря
def preprocess_text(text, dictionary):
    tokens = tokenize(text)
    tokenize_text = ', '.join(tokens)
    cleaned_text = clean_text(tokenize_text)
    lemmatized_text = lemmatize_text(cleaned_text)
    for token in lemmatized_text.split():
        if token not in dictionary:
            dictionary.append(token)

    return lemmatized_text


# Мешок слов
def bag_of_words(text, dct):
    vector = [0] * len(dct)
    token_counts = Counter(text.split())
    for i, word in enumerate(dct):
        vector[i] = token_counts.get(word, 0)
    return vector


# Загрузка TSKV файла с данными
data = pd.read_csv('geo-reviews-dataset-2023.tskv', sep='\t', header=None)

# Установка имен столбцов
data.columns = ['address', 'name_ru', 'rating', 'rubrics', 'text']

# Фильтр только столбцов с текстом отзыва и оценкой
data = data[['text', 'rating']]

# Срез лишнего
data['text'] = data['text'].str.slice(5)
data['rating'] = data['rating'].str.slice(7, -1)
data = data[data['rating'].isin(['1', '2', '3', '4', '5'])]
data['rating'] = data['rating'].astype(int)
data.reset_index(drop=True, inplace=True)


# Снимаем ограничения вывода таблицы
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# Смотрим на данные, выводим 10 первых строк
# print(data[:10])
data = data[:1000]


dictionary = []


# Прогресс-бар для удобства
pb = ProgressBar(total=len(data)-1, prefix='Progress', suffix='Complete', length=50)
print("Прогресс обработки текста")

for i in range(len(data)):
    data.loc[i, 'text'] = preprocess_text(data.loc[i, 'text'], dictionary)
    pb.print_progress_bar(i)


# Разделяем данные на обучающую и тестовую выборки
(train_set, test_set, train_labels, test_labels) = train_test_split(
    data['text'],
    data['rating'],
    test_size=0.3,
    # random_state=42
)

# original_text1 = "Синее небо над головоц. Кошка прыгнула на стол."
# original_text2 = "Синее небо над головой. Кошка прыгнула."
# preprocessed_text = preprocess_text(test_set, dictionary)
# print(preprocessed_text)
# # print(process_text(original_text2, dictionary))
# print(data)
# print(*dictionary)
print(f"Количество слов в словаре: {len(dictionary)}")
# print(*bag_of_words(preprocessed_text, dictionary))
