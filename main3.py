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


def vectorize_text(text, dictionary):
    vector = [0] * len(dictionary)
    for i in range(len(vector)):
        for token in text.split():
            if token == dictionary[i]:
                vector[i] += 1
    return vector


dictionary = []

original_text1 = "Синее небо над головой. Кошка прыгнула на стол."
original_text2 = "Синее небо над головой. Кошка прыгнула."
process_text(original_text1, dictionary)
processed_text = process_text(original_text2, dictionary)
print(processed_text)
# print(process_text(original_text2, dictionary))
print(*dictionary)
print(*vectorize_text(processed_text, dictionary))
