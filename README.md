# Прогнозирование цен автомобилей BMW

## О проекте
Этот проект посвящён **прогнозированию рыночной стоимости автомобилей BMW** на основе технических и эксплуатационных характеристик. Цель — выявить ключевые признаки, влияющие на цену, и построить модели машинного обучения для точного предсказания.

Проект выполнен в **Google Colab** на языке **Python 3**, с использованием библиотек `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`.

---

## Данные
- Датасет: [BMW Car Data Analysis](https://www.kaggle.com/datasets)  
- Основные признаки:
  - `year` — год выпуска
  - `mileage` — пробег
  - `engineSize` — объём двигателя
  - `fuelType` — тип топлива
  - `model` — модель автомобиля
  - `transmission` — тип трансмиссии
  - `price` — цена (целевая переменная)

- Формат данных: CSV
- Признаки как числовые, так и категориальные.

---

## Этапы работы
1. **EDA и очистка данных**
   - Анализ структуры и пропусков (`df.info()`, `df.isna().sum()`)
   - Проверка выбросов и распределений
   - Визуализация корреляций и зависимостей между признаками и ценой
2. **Предобработка**
   - Масштабирование числовых признаков (`StandardScaler`)
   - Кодирование категориальных признаков (`OneHotEncoder`)
   - Построение конвейера `Pipeline` для удобства обучения моделей
3. **Обучение моделей**
   - **Линейная регрессия (Linear Regression)**
   - **Ансамбль деревьев (Random Forest Regressor)**
   - **Градиентный бустинг (XGBoost Regressor)**
4. **Оценка моделей**
   - Метрики: **MAE**, **RMSE**, **R²**
   - Сравнение качества моделей и выбор лучшей
5. **Визуализация результатов**
   - Цена vs пробег
   - Цена vs год выпуска
   - Важность признаков

---

## Метрики качества
- **MAE (Mean Absolute Error)** — средняя абсолютная ошибка предсказаний
- **RMSE (Root Mean Squared Error)** — корень из средней квадратичной ошибки, чувствителен к большим отклонениям
- **R² (Коэффициент детерминации)** — доля объяснённой вариации данных

---

## Используемые библиотеки
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
