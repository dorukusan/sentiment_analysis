from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from console_progressbar import ProgressBar
from sklearn.svm import SVC
from sklearn import metrics
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

# Добавление новых столбцов
data['preprocessed_text'] = None
data['vector'] = None
data['sentiment'] = data['rating']
data['sentiment'] = data['sentiment'].replace({1: -1, 2: -1, 3: 0, 4: 1, 5: 1})


# Снимаем ограничения вывода таблицы
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# Смотрим на данные, выводим 10 первых строк
# print(data[:10])
data = data[:250]


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

print(data[:10])


# Разделяем данные на обучающую и тестовую выборки
(train_set, test_set, train_labels, test_labels) = train_test_split(
    vectors,
    data['sentiment'],
    test_size=0.3,
)


print(test_set)
print(test_labels)


# Обучение модели SVM
svm = SVC(kernel='linear')
svm.fit(train_set, train_labels)

# Предсказание на тестовых данных (SVM)
model_predictions_svm = svm.predict(test_set)

# Оценка точности для SVM
accuracy_svm = metrics.accuracy_score(test_labels, model_predictions_svm)

print("\nОценка точности для SVM:", accuracy_svm)
print(metrics.classification_report(test_labels, model_predictions_svm))

print(*test_labels)
print(*model_predictions_svm)


# print(*dictionary)
# print(f"Количество слов в словаре: {len(dictionary)}")
