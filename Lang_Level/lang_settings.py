import pandas as pd
import numpy as np

import magic
import re
import os
import glob
import time

import pickle

from tqdm.notebook import tqdm

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from gensim.models import Word2Vec


class Text2Vec(TransformerMixin, BaseEstimator):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def fit(self, X, y=None):
        self.w2v = Word2Vec(X, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        return self

    def transform(self, X):
        return np.array([np.mean([self.w2v.wv[w] for w in words if w in self.w2v.wv]
                                 or [np.zeros(self.vector_size)], axis=0) for words in X])
        
        
class TextClassification:
    def __init__(self, dataframe=None, model_path=None):
        if dataframe is not None:
            self.df = dataframe
            self.df['text'] = self.df['text'].apply(lambda s: s.split())
        else:
            self.df = pd.DataFrame()

        self.accuracy = None
        self.report = None

        self.level = None

        if model_path:
            self.load_model(model_path)
            self.is_trained = True
        else:
            self.model = Pipeline([
                ('vec', Text2Vec()),
                ('clf', RandomForestClassifier())
            ])

    def train_model(self):
        X = self.df['text']
        y = self.df['level']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.report = classification_report(y_test, y_pred)
        print(self.report)

        self.is_trained = True
        print('Model trained.')

        # Вычисление точности 
        self.accuracy = accuracy_score(y_test, y_pred)
        print('Model accuracy: ', self.accuracy)


    def train_additional_model(self, additional_df):
        additional_df['text'] = additional_df['text'].apply(lambda s: s.split())
        self.df = pd.concat([self.df, additional_df])
        self.train_model()
        self.is_trained = True
        print('Model trained.')

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        return self.model


    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f'Model saved at {model_path}')

    def get_accuracy(self):
        if self.accuracy:
            print('Accuracy model:', self.accuracy)
            print('')
            print(self.report)
        else:
            print('Model can not has a accuracy, model is empty!')

    def predict(self, text):
        if self.is_trained:
            vectorized_text = [text.split()] 
            self.level = self.model.predict(vectorized_text)[0]
            probabilities = self.model.predict_proba(vectorized_text)[0]
            levels = self.model.classes_
            percentage = dict(zip(levels, [f"{p * 100:.2f}%" for p in probabilities]))
            return percentage
        else:
            print('Model is not trained yet.')
            return None

class TextOperations:
    def __init__(self, model=None):
        self.level_mapping = {'A1': 0, 'A2': 1, 'B1': 2, 'B2' : 3, 'C1' : 4, 'C2' :5}

        if model:
            self.tc = TextClassification(model_path=model)

    def read_data(self, file):
        blob = open(file, 'rb').read()
        m = magic.Magic(mime_encoding=True)
        encoding = m.from_buffer(blob)

        if not encoding in ['unknown-8bit', 'binary']:
            with open(file, 'r', encoding=encoding, errors='ignore') as f:
                text = f.read().strip()
            return text
        else:
            return ''

    def clean_data(self, subs):
        pattern = r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\s*(.*)'

        # Находим все подходящие под образец строки и извлекаем текст
        matches = re.findall(pattern, subs, re.DOTALL)

        # Обрабатываем текст: приводим к нижнему регистру, удаляем стоп-слова, проводим лемматизацию
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        cleaned_matches = []
        for match in matches:
            words = match.split()
            words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in stop_words and word.isalpha()]
            cleaned_matches.append(' '.join(words))

        return pd.DataFrame(cleaned_matches, columns=['text'])


    def load_data(self, subtitle_path, level=''):
        text = self.read_data(subtitle_path)

        if text != '':
            df = self.clean_data(text)

            df['file'] = subtitle_path

            if level != '':
                df["level"] = level
                return df.reindex(columns=['file', 'text', 'level'])
            else:
                return df.reindex(columns=['file', 'text'])
        else:
            return None

    def multi_load_data(self, paths, levels):
        dataframes = pd.DataFrame(columns=["file", "text", "level"])
        for idx, file in enumerate(paths):
            df = self.load_data(file, levels[idx])
            
            if df is not None:
                dataframes = pd.concat([dataframes, df], ignore_index=True)

        return dataframes

    def load_data_from_directory(self, list, directory_path):
        # Чтение исходного файла
        df = pd.read_excel(list)

        # Делаем отдельный список с названиями фильмов
        movies = df['Movie'].str.replace('_', ' ').tolist()

        # Перебираем файлы в директории и сравнивают каждый файл с каждым названием фильма из списка
        data = []
        for filename in glob.glob(os.path.join(directory_path, '*.srt')):
            movie_name = re.sub('(_|-|\.)', ' ', filename) # Заменяем символы на пробелы для упрощения сравнения
            best_match = process.extractOne(movie_name, movies) # Находим наилучшее совпадение
            similarity = best_match[1] # Вытаскиваем процент совпадения

            # Если процент совпадения выше некоторого порога, добавляем файл в список
            if similarity > 80: # Можно подобрать оптимальный порог в зависимости от данных
                idx = df[df['Movie'].str.replace('_', ' ') == best_match[0]].index.values.astype(int)[0] # Находим индекс наилучшего совпадения
                level = df.loc[idx, 'Level'] # Находим соответствующий уровень
                data.append({"file": filename, "movie": best_match[0], "level": level})

        df_files = pd.DataFrame(data)

        dataframes = pd.DataFrame(columns=["file", "text", "level"])

        t = tqdm(total=len(df_files)) 

        for index, row in (df_files.iterrows()):
            df = self.load_data(row['file'], row['level'])

            if df is not None:
                dataframes = pd.concat([dataframes, df], ignore_index=True)

            time.sleep(0.1)
            t.update(1)

        t.close()

        return dataframes

    def predict_level(self, data):
        #sFile = file
        #df = self.load_data(sFile)
        #text = df.at[0,"text"]

        #text = ['this very difficult text'.split()]
        predictions = self.tc.predict(data)
        
        prediction_df = pd.Series(predictions).to_frame()
        prediction_df.columns = ['level']

        return self.tc.level, prediction_df