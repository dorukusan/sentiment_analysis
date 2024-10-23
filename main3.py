from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from console_progressbar import ProgressBar
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import re
import spacy
import pandas as pd
import numpy as np


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

# Добавление новых столбцов
data['preprocessed_text'] = None
data['vector'] = None
data['sentiment'] = data['rating']
data['sentiment'] = data['sentiment'].replace({1: -1, 2: 0, 3: 0, 4: 0, 5: 1})
counts = data['sentiment'].value_counts()
print(counts)


# Снимаем ограничения вывода таблицы
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# Смотрим на данные, выводим 10 первых строк
# print(data[:10])
# data = data[:1000]
n = 1500
data = pd.concat([
    data[data['sentiment'] == -1].sample(n=n, random_state=1),
    data[data['sentiment'] == 0].sample(n=n, random_state=1),
    data[data['sentiment'] == 1].sample(n=n, random_state=1)])

# counts = data['sentiment'].value_counts()
# print(counts)
# print(data)
data.reset_index(drop=True, inplace=True)


dictionary = []


# Прогресс-бар для наглядности прогресса обработки текста
pb = ProgressBar(total=len(data)-1, prefix='Progress', suffix='Complete', length=50)
print("Прогресс обработки текста")

# Предварительная обработка текста
for i in range(len(data)):
    data.loc[i, 'preprocessed_text'] = preprocess_text(data.loc[i, 'text'], dictionary)
    pb.print_progress_bar(i)


# Прогресс-бар для наглядности прогресса векторизации текста
pb = ProgressBar(total=len(data)-1, prefix='Progress', suffix='Complete', length=50)
print("\nПрогресс векторизации текста")


vectors = []

# Векторизация текста
for i in range(len(data)):
    vectors.append(bag_of_words(data.loc[i, 'preprocessed_text'], dictionary))
    data.loc[i, 'vector'] = str(bag_of_words(data.loc[i, 'preprocessed_text'], dictionary))
    pb.print_progress_bar(i)


# print(*dictionary)
# print(f"Количество слов в словаре: {len(dictionary)}")


# Разделяем данные на обучающую и тестовую выборки
(train_set, test_set, train_labels, test_labels) = train_test_split(
    vectors,
    data['sentiment'],
    test_size=0.3,
    random_state=42
)


def model_svm(train_x, test_x, train_y, test_y):
    # Обучение модели SVM
    svm = SVC(kernel='linear')
    svm.fit(train_x, train_y)

    # Предсказание на тестовых данных (SVM)
    model_predictions_svm = svm.predict(test_x)

    # Оценка точности для SVM
    accuracy_svm = metrics.accuracy_score(test_y, model_predictions_svm)
    print("\nОценка точности для SVM:", accuracy_svm)
    print(metrics.classification_report(test_y, model_predictions_svm))


def model_kmeans(train_x, test_x, test_y):
    # Обучение модели KMeans с использованием инициализации k-means++
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=1000, n_init=10, random_state=42)
    kmeans.fit(train_x)

    # Предсказание на тестовых данных (KMeans)
    model_predictions_kmeans = kmeans.predict(test_x)

    # Создаем DataFrame для удобства
    results = pd.DataFrame({'Cluster': model_predictions_kmeans, 'TrueLabel': test_y})

    # Находим наиболее частую метку для каждого кластера
    cluster_labels = results.groupby('Cluster')['TrueLabel'].agg(lambda x: x.mode()[0]).to_dict()
    print("\nСоответствие кластеров и истинных меток:")
    print(cluster_labels)

    # Сопоставление кластеров с истинными метками
    mapped_predictions_kmeans = np.array([cluster_labels[cluster] for cluster in model_predictions_kmeans])

    # Оценка точности для KMeans
    accuracy_kmeans = metrics.accuracy_score(test_y, mapped_predictions_kmeans)
    print("\nОценка точности для KMeans:", accuracy_kmeans)
    print(metrics.classification_report(test_y, mapped_predictions_kmeans))


def model_gnb(train_x, test_x, train_y, test_y):
    # Обучение модели GaussianNB
    gnb = GaussianNB()
    gnb.fit(train_x, train_y)

    # Предсказание на тестовых данных (GaussianNB)
    model_predictions_gnb = gnb.predict(test_x)

    # Оценка точности для GaussianNB
    accuracy_kmeans = metrics.accuracy_score(test_y, model_predictions_gnb)
    print("\nОценка точности для GaussianNB:", accuracy_kmeans)
    print(metrics.classification_report(test_y, model_predictions_gnb))


model_svm(train_set, test_set, train_labels, test_labels)
model_kmeans(train_set, test_set, test_labels)
model_gnb(train_set, test_set, train_labels, test_labels)
