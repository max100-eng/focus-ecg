import streamlit as st
import requests
import json

# La URL CORRECTA de tu API de Hugging Face
API_URL = "https://maxxxi100-mi-api-ecg.hf.space/plumber/prediccion_ecg"

st.title("Aplicación de Predicción de ECG")
st.write("Introduce los datos del paciente para obtener una predicción.")

# Campos de entrada para los datos
frecuencia_cardiaca = st.number_input("Frecuencia Cardíaca", min_value=0, max_value=200, value=75)
variabilidad_rr = st.number_input("Variabilidad RR", min_value=0, max_value=200, value=120)

if st.button("Obtener Predicción"):
    # Prepara los datos para la solicitud POST
    datos_a_enviar = {
        "datos": [
            {
                "frecuencia_cardiaca": frecuencia_cardiaca,
                "variabilidad_rr": variabilidad_rr
            }
        ]
    }

    try:
        # Envía la solicitud POST a tu API
        response = requests.post(API_URL, json=datos_a_enviar)

        # Si la solicitud fue exitosa (código 200)
        if response.status_code == 200:
            prediccion = response.json()
            st.success(f"La predicción es: {prediccion['prediccion']}")
        else:
            st.error(f"Error al conectar con la API: {response.status_code}")
            st.error(f"Respuesta del servidor: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión: {e}") 