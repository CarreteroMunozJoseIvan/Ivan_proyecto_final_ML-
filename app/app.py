import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn import preprocessing
# from src.model import*  ####ModuleNotFoundError: No module named 'model'

import sys
sys.path.append('../src/model.py')


import pickle

df=pd.read_csv("./data/processed_data.csv")
df=df.drop('id',axis=1)

df2=df.copy()
df.head()
df.drop(columns='posting_date', inplace=True)
html_temp = """
<div style="background-color:yellow;padding:1.5px">
<h1 style="color:black;text-align:center;">Used Car Price Prediction</h1>
</div><br>"""
st.markdown(html_temp, unsafe_allow_html=True)


st.write("\n\n"*2)

file_path = os.path.join("models", "train_model.pkl")
label_encoder_path = os.path.join("models", "label_encoder.pkl")

# filename = 'train_model.pkl'
ruta_train_model = './models/train_model.pkl'

with open(ruta_train_model, 'rb') as archivo:
    train_model_final = pickle.load(archivo)

ruta_label_encoder = './models/label_encoder.pkl'

with open(ruta_label_encoder, 'rb') as archivo:
    label_encoder_final = pickle.load(archivo)

with st.sidebar:
    st.subheader('Car Specs to Predict Price')
    import streamlit as st


# Supongamos que tienes una lista de columnas categóricas llamada 'cat_cols'
num_col=['year','odometer','lat','long']
cat_cols=['region','manufacturer','model','condition','cylinders','fuel','title_status', 'transmission', 'drive', 'size', 'type', 'paint_color' ]

# Crea una interfaz en Streamlit para seleccionar las categorías
selected_categories = {}
opciones_categorias = {}

for col in cat_cols:
    # Aplica LabelEncoder a la columna categórica
    df[col] = label_encoder_final.fit_transform(df[col])

    # Obtiene las categorías y sus números asignados
    categories = label_encoder_final.classes_
    category_nums = label_encoder_final.transform(categories)
    
   
    selected_category = st.selectbox(f'Selecciona una categoría para {col}', opciones_categorias[col].keys())
    selected_categories[col] = opciones_categorias[col][selected_category]

    # Asigna las categorías y sus números al diccionario opciones_categorias
    for category, num in zip(categories, category_nums):
        opciones_categorias[col][category] = num

    # Imprime las categorías y sus números asignados
    print(f"Columna: {col}")
    for category, num in opciones_categorias[col].items():
        print(f"Categoría '{category}': {num}")

encoded_inputs = [selected_categories[col] for col in cat_cols]    
prediccion = train_model_final.predict([encoded_inputs])
