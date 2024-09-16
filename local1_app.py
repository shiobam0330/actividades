import pandas as pd
import numpy as np
import pickle
import streamlit as st
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model



path = "D:/Downloads/INTELIGENCIA ARTIFICIAL/actividad individual/"
dataset = pd.read_csv(path + "dataset_APP.csv",header = 0,sep=";",decimal=",")

with open(path +'best_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

cuantitativas = ['Avg. Session Length','Time on App',
                 'Time on Website', 'Length of Membership']
categoricas = ['dominio', 'Tec']
base_modelo = pd.concat([dataset.get(cuantitativas),dataset.get(categoricas)],axis = 1)
base_modelo["y"] = dataset["price"].copy()


st.title("API de Prediccion de Precio")


Avg = st.text_input("Avg Session Length", value="32.063775")	
Time_App = st.text_input("Time on App", value="10.71915")	
Time_Website = st.text_input("Time on Website", value="37.712509")
Length_Membership = st.text_input("Length of Membership", value="3.004743")
dominio = st.selectbox("Seleccione el dominio:", ['gmail', 'Otro', 'hotmail', 'yahoo'])
Tec = st.selectbox("Seleccione el tipo de dispositivo:", ['Smartphone', 'Portatil', 'PC', 'Iphone'])

if st.button("Calcular"):
    try:
        Avg = float(Avg)
        Time_App = float(Time_App)
        Time_Website = float(Time_Website)
        Length_Membership = float(Length_Membership)


        user = pd.DataFrame({'x1':[Avg],'x2': [Time_App],'x3': [Time_Website],
                             'x4': [Length_Membership], 'x5': [dominio], 'x6':[Tec]})
        base_modelo = base_modelo.drop('y', axis=1)
        user.columns = base_modelo.columns

        prueba2 = pd.concat([user, base_modelo], axis=0)
        prueba2.index = range(prueba2.shape[0])

        cuantitativas = ['Avg. Session Length','Time on App',
                         'Time on Website', 'Length of Membership']
        
        categoricas = ['dominio', 'Tec']

        predictions = predict_model(modelo, data=prueba2)
        predictions["y"] = predictions["prediction_label"].map(float)

        st.write(f'La predicción es: {predictions.iloc[0]["y"]}')

    except ValueError:
        st.error("Por favor, ingrese valores numéricos válidos en los campos correspondientes.")

# Botón para reiniciar la consulta
if st.button("Reiniciar"):
    st.experimental_rerun()


