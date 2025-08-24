# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import json
import random

# Streamlit page configuration (title, layout, and custom theme)
st.set_page_config(
    page_title="Focus ECG",
    page_icon="❤️",
    layout="wide"
)

# Custom theme with JavaScript and updated CSS classes
custom_theme_script = """
<style>
    /* Estilos del tema oscuro */
    body {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stApp {
        background-color: #0E1117;
    }
    .st-emotion-cache-1cpx96c { /* Sidebar */
        background-color: #262730;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
    }
    .st-emotion-cache-12fmw6v, .st-emotion-cache-1r6chqg {
        background-color: #0E1117;
    }
    /* Ocultar el menú de Streamlit y el pie de página */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

st.markdown(custom_theme_script, unsafe_allow_html=True)


# Título de la aplicación
st.title("❤️ Focus ECG")
st.markdown("---")

# --- FUNCIONES DE ANÁLISIS ---

# Función para cargar y predecir con el modelo de TensorFlow
@st.cache_resource
def load_ecg_model():
    """
    Carga el modelo de IA una sola vez.
    Asegúrate de que tu archivo de modelo ('modelo_ecg.h5') esté en la misma carpeta.
    """
    try:
        # Aquí se carga el modelo de TensorFlow
        model = keras.models.load_model('modelo_ecg.h5')
        st.info("Modelo de TensorFlow cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def analyze_ecg_details(ecg_signal):
    """
    Simula un análisis detallado de los elementos del ECG basado en datos numéricos.
    Esta función es solo para fines de demostración.
    """
    # Valores aleatorios que simulan un análisis del modelo
    pr_interval = random.uniform(0.12, 0.22)
    qrs_duration = random.uniform(0.06, 0.15)
    st_segment = random.uniform(-0.1, 0.2)
    qt_interval = random.uniform(0.35, 0.50)
    
    # Simulación de la forma de las ondas
    onda_q_profunda = random.choice([True, False])
    st_supradesnivel = st_segment > 0.1
    st_infradesnivel = st_segment < -0.05
    eje_desviado_derecha = random.choice([True, False])
    eje_desviado_izquierda = random.choice([True, False])

    # Construir el reporte detallado
    reporte = {
        "Frecuencia Cardíaca (lpm)": random.randint(60, 100),
        "Ritmo": "Regular" if random.random() > 0.1 else "Irregular",
        "Onda P": "Presente y normal",
        "Intervalo PR (s)": f"{pr_interval:.2f} ({'Normal' if 0.12 <= pr_interval <= 0.20 else 'Alargado'})",
        "Duración QRS (s)": f"{qrs_duration:.2f} ({'Normal' if qrs_duration <= 0.12 else 'Ancho'})",
        "Segmento ST": f"{st_segment:.2f} mV ({'Supradesnivel' if st_supradesnivel else ('Infradesnivel' if st_infradesnivel else 'Isoeléctrico')})",
        "Onda Q": "Normal" if not onda_q_profunda else "Patológica",
        "Onda T": "Normal",
        "Intervalo QT (s)": f"{qt_interval:.2f} ({'Normal' if qt_interval < 0.45 else 'Alargado'})",
        "Eje Cardíaco": "Normal"
    }

    if eje_desviado_derecha:
        reporte["Eje Cardíaco"] = "Desviado a la derecha"
    elif eje_desviado_izquierda:
        reporte["Eje Cardíaco"] = "Desviado a la izquierda"

    # Determinar el diagnóstico final basado en las simulaciones
    if "Supradesnivel" in reporte["Segmento ST"] and reporte["Onda Q"] == "Patológica":
        diagnostico_final = "Infarto Agudo del Miocardio (IAM)"
    elif "Infradesnivel" in reporte["Segmento ST"]:
        diagnostico_final = "Angina de pecho"
    elif "Ancho" in reporte["Duración QRS (s)"]:
        diagnostico_final = "Bloqueo de Branca"
    elif "Alargado" in reporte["Intervalo PR (s)"]:
        diagnostico_final = "Bloqueo del Seno Atrial"
    elif reporte["Ritmo"] == "Irregular":
        diagnostico_final = "Arritmia"
    else:
        diagnostico_final = "Ritmo sinusal normal"

    return {"diagnostico": diagnostico_final, "analisis_detallado": reporte}

def predict_with_model(data, model, file_type):
    """
    Realiza una predicción sobre los datos ECG usando el modelo.
    """
    if model:
        st.info("Modelo cargado. Preprocesando y prediciendo...")
        try:
            # Si el archivo es una imagen, se usan datos de ejemplo
            if file_type in ["image/png", "image/jpeg", "image/jpg"]:
                data_numpy = np.random.randn(1000)
            elif isinstance(data, pd.DataFrame):
                # Suponiendo que la columna con la señal ECG se llama 'ECG_signal'
                if 'ECG_signal' in data.columns:
                    data_numpy = data['ECG_signal'].values
                else:
                    st.error("Columna 'ECG_signal' no encontrada en el archivo CSV.")
                    return None
            else:
                data_numpy = np.array(data)

            # --- PASO 1: Preprocesamiento de los datos ---
            required_shape = model.input_shape[1:]
            
            if len(data_numpy) > required_shape[0]:
                data_processed = data_numpy[:required_shape[0]]
            elif len(data_numpy) < required_shape[0]:
                padding = np.zeros(required_shape[0] - len(data_numpy))
                data_processed = np.concatenate((data_numpy, padding))
            else:
                data_processed = data_numpy

            data_processed = (data_processed - np.mean(data_processed)) / np.std(data_processed)
            data_processed = data_processed.reshape(1, *required_shape)

            # --- PASO 2: Predicción ---
            # La predicción del modelo real se usaría aquí para un diagnóstico.
            # prediction = model.predict(data_processed)
            # st.write("Resultado de la predicción (probabilidades):", prediction)
            
            # --- Simulación de un análisis detallado ---
            return analyze_ecg_details(data_processed)

        except Exception as e:
            st.error(f"Error durante la predicción con el modelo: {e}")
            return None
    
    else:
        st.warning("El modelo no ha podido ser cargado. No se puede realizar la predicción.")
        return None

# Cargar el modelo de IA al iniciar la aplicación
ecg_model = load_ecg_model()

# --- DISEÑO DE LA APLICACIÓN DE UNA SOLA PÁGINA ---

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("Análisis de ECG")
    st.write("Sube una imagen o usa la cámara para analizar un electrocardiograma.")
    st.write("La IA te proporcionará un resumen detallado y las mediciones principales.")
    
    st.markdown("""
        <div style="
            background-color: #362f1c;
            border-left: 5px solid #ffcc00;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
        ">
        <h5 style="color: #ffcc00; margin: 0;">AVISO IMPORTANTE:</h5>
        <p style="color: #ffcc00; margin-top: 5px;">
        Este análisis es **solo para fines informativos** y no constituye un diagnóstico médico.
        Siempre consulta a un profesional de la salud calificado para una interpretación precisa
        de cualquier dato médico.
        </p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Subir ECG")
    archivo = st.file_uploader(
        "Selecciona un archivo ECG (formatos admitidos: CSV, TXT, PNG, JPG, JPEG)",
        type=['csv', 'txt', 'png', 'jpg', 'jpeg']
    )

    if archivo is not None:
        st.success(f"Archivo {archivo.name} subido exitosamente!")
        
        # Guardar el archivo subido en el estado de la sesión
        st.session_state['last_uploaded_file'] = archivo
        st.session_state['last_uploaded_file_type'] = archivo.type

        with st.spinner("Procesando señal ECG..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            
            try:
                file_type = archivo.type
                data = None
                
                if file_type in ["text/csv", "text/plain"]:
                    data = pd.read_csv(archivo)
                elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                    # Los datos de la imagen se simulan para la predicción
                    data = np.random.randn(1000)
                else:
                    st.warning("Tipo de archivo no soportado para análisis.")
                    data = None

                if data is not None:
                    results = predict_with_model(data, ecg_model, file_type)
                    
                    if results:
                        st.session_state['results'] = results
                        st.session_state['processed'] = True
                        st.success("Procesamiento completado!")
                    else:
                        st.session_state['processed'] = False
                else:
                    st.session_state['processed'] = False
            except Exception as e:
                st.error(f"Ocurrió un error durante el análisis: {e}")
                st.session_state['processed'] = False

with col2:
    if 'processed' in st.session_state and st.session_state['processed']:
        st.subheader("Resultados del análisis:")
        results = st.session_state['results']

        if 'last_uploaded_file_type' in st.session_state and st.session_state['last_uploaded_file_type'] in ["image/png", "image/jpeg", "image/jpg"]:
            # Display the uploaded image
            st.subheader("ECG Subido")
            st.image(st.session_state['last_uploaded_file'], caption="ECG Subido")
        
        st.subheader("Diagnóstico")
        diagnostico = results['diagnostico']
        
        if "normal" in diagnostico.lower():
            st.success(diagnostico)
        else:
            st.error(diagnostico)
            
        st.subheader("Análisis Detallado de Elementos del ECG")
        
        # Mostrar el reporte detallado en una tabla
        analisis_df = pd.DataFrame(results['analisis_detallado'].items(), columns=['Elemento', 'Estado'])
        st.table(analisis_df)
    else:
        st.subheader("Resultados del análisis:")
        st.warning("Por favor, sube y procesa un archivo ECG para ver el informe.")
