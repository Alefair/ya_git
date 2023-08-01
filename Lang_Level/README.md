# Сервис по определению уровня английского языка по файлу субтитра

![logo](https://raw.githubusercontent.com/Alefair/ya_git/master/Lang_Level/langLogo.png)

**Проект**
```
Идея основана на том, что просмотр фильмов в оригинале является популярным и эффективным методом
изучения иностранных языков.

Ключевой момент заключается в выборе фильма, который бы был подходящим для студента с точки зрения
сложности, то есть студент мог бы понять 80-85% диалогов.

Для этого преподаватель должен просмотреть фильм и определить его уровень сложности, что влечет
за собой значительные временные затраты.
```

**Задача заказчика**
```
Создать ML решение для автоматизированного определения сложности англоязычных фильмов и
развернуть приложение для демонстрации его работы, при условии возможности
использования пакета Streamlit.
```

******************

Тестовый стенд расположен [здесь](https://lang.alefair.com)

### Способ использования:

- Скопировать к себе в проект папки:
  - models
  - subtitles

- И файл lang_settings.py

- Создаем тетрадку и пишем код:

**Импортируем библиотеки**
```python
from lang_settings import TextOperations, TextClassification, Text2Vec
import os
import pandas as pd
```

**Для обучения**
  ```python
# Указываем путь к файлу размеченных Фильм = level (файл movies_labels.xlsx)
# Указываем путь к папке с субтитрами

df_train = TextOperations().load_data_from_directory('./subtitles/movies_labels.xlsx', './subtitles')
df_train
  ```
![table](https://raw.githubusercontent.com/Alefair/ya_git/master/Lang_Level/src/table.png)

```python
# Обучаем
tc = TextClassification(df_train)
tc.train_model()
```
![accuracy](https://raw.githubusercontent.com/Alefair/ya_git/master/Lang_Level/src/accuracy.png)

```python
# сохраним модель
tc.save_model('/content/model.pkl')
```

**Используем**
  ```python
# Указываем путь к модели
model = TextOperations('./models/model.pkl')

# Указываем путь к субтитру для определения уровня английского языка
df = model.load_data('/home/alefair/jupyter/work/app/subtitles/Frozen.2013.WEB-DL.DSNP.srt')
text = df.at[0,"text"]

# Предсказываем
level, dt = model.predict_level(text)

print(level)
display(dt)
```

![predict](https://raw.githubusercontent.com/Alefair/ya_git/master/Lang_Level/src/predict.png)
