import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Prediction", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "linear_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

# Загружаем модель

MODEL, FEATURE_NAMES = load_model()

st.title("Предсказание стоимости автомобиля")

uploaded_file = st.file_uploader("Загрузите CSV с данными автомобилей", type=["csv"])
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

# Предсказания для всех автомобилей

features = df[FEATURE_NAMES]
predictions = MODEL.predict(features)
df['prediction'] = predictions
st.subheader("Данные с предсказаниями")
st.write(df)

# Форма для одного автомобиля

st.subheader("Предсказания для одного авто")
with st.form("single"):
    year = st.number_input("year", value=2020)
    km_driven = st.number_input("km_driven", value=0)
    mileage = st.number_input("mileage", value=0.0)
    engine = st.number_input("engine", value=0.0)
    max_power = st.number_input("max_power", value=0.0)
    seats = st.number_input("seats", value=5)
    submitted = st.form_submit_button("Предсказать")

if submitted:
    input_df = pd.DataFrame([{
        'year': year,
        'km_driven': km_driven,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats
    }])
    pred = MODEL.predict(input_df[FEATURE_NAMES])[0]
    st.write("Предсказание:", pred)

# EDA визуализация

st.subheader("EDA визуализация")

st.write("Попарные распределения числовых признаков")
pair = sns.pairplot(df[FEATURE_NAMES])
st.pyplot(pair.fig)

st.write("Рспределение данных признаков")
fig, ax = plt.subplots(figsize=(10, 6))
df[FEATURE_NAMES].boxplot(ax=ax)
st.pyplot(fig)

# Визуализация весов модели

st.subheader("Веса модели")

df = pd.DataFrame({
    'feature': FEATURE_NAMES,
    'weight': MODEL.coef_
})

st.bar_chart(df.set_index('feature'))
