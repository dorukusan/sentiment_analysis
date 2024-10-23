from nltk.corpus import stopwords
from collections import Counter
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from console_progressbar import ProgressBar
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import re
import spacy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import string


nlp_rus = spacy.load("ru_core_news_sm")
nlp_eng = spacy.load("en_core_web_sm")


# Токенизация текста
def tokenize(text):
    text = text.lower()
    tokens = text.split()
    return tokens


# Чистка текста
def clean_text(text, lang):
    text = text.replace("\\n", " ")
    text = text.replace("< br/>", " ")
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    if lang == "rus":
        stop_words = set(stopwords.words('russian'))
    filtered_words = [word for word in text.split() if word not in stop_words]
    return " ".join(filtered_words)


# Лемматизация текста
def lemmatize_text(cleaned_text, lang):
    doc = nlp_eng(cleaned_text)
    if lang == 'rus':
        doc = nlp_rus(cleaned_text)
    return " ".join([token.lemma_ for token in doc])


# Предварительная обработка текста и заполнение словаря
def preprocess_text(text, dct, lang):
    tokens = tokenize(text)
    tokenize_text = ', '.join(tokens)
    cleaned_text = clean_text(tokenize_text, lang)
    lemmatized_text = lemmatize_text(cleaned_text, lang)
    for token in lemmatized_text.split():
        if token not in dct:
            dct.append(token)

    return lemmatized_text


# Мешок слов
def bag_of_words(text, dct):
    vector = [0] * len(dct)
    token_counts = Counter(text.split())
    for i, word in enumerate(dct):
        vector[i] = token_counts.get(word, 0)
    return vector


def model_Word2Vec(data, vectors_w2v):
    sentences = [sentence.split() for sentence in data['preprocessed_text']]
    w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
    for sentence in sentences:
        words_vecs = [w2v.wv[word] for word in sentence if word in w2v.wv]
        if len(words_vecs) == 0:
            vectors_w2v.append(np.zeros(100))
            continue
        words_vecs = np.array(words_vecs)
        vectors_w2v.append(words_vecs.mean(axis=0))


# Предобработка датасета
def preprocessing_data(data, language):
    dataset = data.copy()
    if language == "rus":
        dataset = dataset[['text', 'rating']]  # Фильтр только столбцов с текстом отзыва и оценкой
        dataset['text'] = dataset['text'].str.slice(5)  # Срез лишнего
        dataset['rating'] = dataset['rating'].str.slice(7, -1)
        dataset = dataset[dataset['rating'].isin(['1', '3', '5'])]
        # data['sentiment'] = data['rating']
        dataset['sentiment'] = dataset['rating'].replace({'1': 0, '3': 1, '5': 2})
    else:
        dataset = dataset[['text', 'sentiment']]
        dataset['sentiment'] = dataset['sentiment'].replace({'negative': 0, 'neutral': 1, 'positive': 2})
    dataset['preprocessed_text'] = None
    dataset['vector'] = None
    dataset.reset_index(drop=True, inplace=True)
    counts = dataset['sentiment'].value_counts()
    print(counts)

    # Снимаем ограничения вывода таблицы
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    print(dataset[:10])

    return dataset


# Консольное меню
print("МЕНЮ\n\n[1] Отзывы с Яндекс.Карт\n[2] Маленький синтетический датасет на английском (pos, neu, neg)\n"
      "[3] Твиты про авиакомпанию (pos, neu, neg)\n[4] Отзывы на фильмы (только pos и neg)\n[0] ВЫХОД")

lang = "eng"

data = pd.read_csv('Tweets.csv')
data.rename(columns={'airline_sentiment': 'sentiment'}, inplace=True)
data = preprocessing_data(data, lang)

# # Загрузка файла с данными
# while True:
#     menu = int(input("\nВыберите датасет: "))
#     n = 1000
#     lang = "eng"
#     if menu == 1:
#         data = pd.read_csv('geo-reviews-dataset-2023.tskv', sep='\t')
#         data.columns = ['address', 'name_ru', 'rating', 'rubrics', 'text']
#         lang = "rus"
#         data = preprocessing_data(data, lang)
#
#         data = pd.concat([
#             data[data['sentiment'] == 0].sample(n=n, random_state=1),
#             data[data['sentiment'] == 1].sample(n=n, random_state=1),
#             data[data['sentiment'] == 2].sample(n=n, random_state=1)])
#         data.reset_index(drop=True, inplace=True)
#
#     elif menu == 2:
#         data = pd.read_csv('sentiment_analysis.csv')
#         data = preprocessing_data(data, lang)
#
#     elif menu == 3:
#         data = pd.read_csv('Tweets.csv')
#         data.rename(columns={'airline_sentiment': 'sentiment'}, inplace=True)
#         data = preprocessing_data(data, lang)
#
#     elif menu == 4:
#         data = pd.read_csv('IMDB-Dataset.csv')
#         data.rename(columns={'review': 'text'}, inplace=True)
#         data = preprocessing_data(data, lang)
#         data = pd.concat([
#             data[data['sentiment'] == 0].sample(n=n, random_state=1),
#             data[data['sentiment'] == ].sample(n=n, random_state=1)])
#         data.reset_index(drop=True, inplace=True)
#
#     elif menu == 0:
#         print("\nВЫХОД ИЗ ПРОГРАММЫ")
#         # exit(0)
#         continue
#     else:
#         print("Выберите существующий датасет!")


dictionary = []


# Прогресс-бар для наглядности прогресса обработки текста
pb = ProgressBar(total=len(data)-1, prefix='Progress', suffix='Complete', length=50)
print("Прогресс обработки текста")

# Предварительная обработка текста
for i in range(len(data)):
    data.loc[i, 'preprocessed_text'] = preprocess_text(data.loc[i, 'text'], dictionary, lang)
    pb.print_progress_bar(i)


# Прогресс-бар для наглядности прогресса векторизации текста
pb = ProgressBar(total=len(data)-1, prefix='Progress', suffix='Complete', length=50)
print("\nПрогресс векторизации текста")


vectors_bow = []


# Векторизация текста
for i in range(len(data)):
    vectors_bow.append(bag_of_words(data.loc[i, 'preprocessed_text'], dictionary))
    data.loc[i, 'vector'] = str(bag_of_words(data.loc[i, 'preprocessed_text'], dictionary))
    pb.print_progress_bar(i)


# print(*dictionary)
# print(f"Количество слов в словаре: {len(dictionary)}")


vectors_w2v = []
model_Word2Vec(data, vectors_w2v)


# Разделяем данные на обучающую и тестовую выборки для машка слов
(train_set_bow, test_set_bow, train_labels_bow, test_labels_bow) = train_test_split(
    vectors_bow,
    data['sentiment'],
    test_size=0.3,
    random_state=42
)


# Разделяем данные на обучающую и тестовую выборки для Word2Vec
(train_set_w2v, test_set_w2v, train_labels_w2v, test_labels_w2v) = train_test_split(
    vectors_w2v,
    data['sentiment'],
    test_size=0.3,
    random_state=42
)


# Метод опорных векторов
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


# Логистическая регрессия
def model_logistic_regression(train_x, test_x, train_y, test_y):
    # Обучение модели Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(train_x, train_y)

    # Предсказание на тестовых данных (Logistic Regression)
    model_predictions_logreg = logreg.predict(test_x)

    # Оценка точности для Logistic Regression
    accuracy_logreg = metrics.accuracy_score(test_y, model_predictions_logreg)
    print("\nОценка точности для Logistic Regression:", accuracy_logreg)
    print(metrics.classification_report(test_y, model_predictions_logreg))


# Гауссовский наивный байесовский классификатор
def model_gnb(train_x, test_x, train_y, test_y):
    # Обучение модели GaussianNB
    gnb = GaussianNB()
    gnb.fit(train_x, train_y)

    # Предсказание на тестовых данных (GaussianNB)
    model_predictions_gnb = gnb.predict(test_x)

    # Оценка точности для GaussianNB
    accuracy_gnb = metrics.accuracy_score(test_y, model_predictions_gnb)
    print("\nОценка точности для GaussianNB:", accuracy_gnb)
    print(metrics.classification_report(test_y, model_predictions_gnb))


# print("Обучение моделей с использованием Bag-of-words")
# model_svm(train_set_bow, test_set_bow, train_labels_bow, test_labels_bow)
# model_logistic_regression(train_set_bow, test_set_bow, train_labels_bow, test_labels_bow)
# model_gnb(train_set_bow, test_set_bow, train_labels_bow, test_labels_bow)
#
#
# print("Обучение моделей с использованием Word2Vec")
# model_svm(train_set_w2v, test_set_w2v, train_labels_w2v, test_labels_w2v)
# model_logistic_regression(train_set_w2v, test_set_w2v, train_labels_w2v, test_labels_w2v)
# model_gnb(train_set_w2v, test_set_w2v, train_labels_w2v, test_labels_w2v)


train_set_w2v = np.array(train_set_w2v)
test_set_w2v = np.array(test_set_w2v)


scaler = StandardScaler()
X_train = scaler.fit_transform(train_set_w2v)
X_test = scaler.transform(test_set_w2v)
X_train_tensor = torch.FloatTensor(train_set_w2v)
Y_train_tensor = torch.LongTensor(train_labels_w2v.values)
X_test_tensor = torch.FloatTensor(test_set_w2v)
Y_test_tensor = torch.LongTensor(test_labels_w2v.values)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 200)
        self.fc2 = nn.Linear(200, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()  # Обнуляем градиенты

    outputs = model(X_train_tensor)  # Прямой проход
    loss = criterion(outputs, Y_train_tensor)  # Вычисление потерь
    loss.backward()  # Обратное распространение ошибок
    optimizer.step()  # Обновление весов

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)

accuracy = (predicted == Y_test_tensor).sum().item() / Y_test_tensor.size(0)
print(f'Accuracy: {accuracy:.4f}')
