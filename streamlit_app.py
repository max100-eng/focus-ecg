import streamlit as st
import requests
import numpy as np
import pandas as pd
from generate_ecg import generate_ecg # Importamos la función de simulación

# --- CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(page_title="Focus-ECG Predictor & Simulator", layout="wide")

st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #007bff;
    }
    /* Estilo para separar secciones */
    .section-divider {
        margin-top: 20px;
        margin-bottom: 20px;
        border-top: 2px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Focus-ECG: Análisis, Predicción y Simulación</p>', unsafe_allow_html=True)


# --- CONFIGURACIÓN DE LA API DE R ---
# NOTA IMPORTANTE: Debes reemplazar esta URL por tu URL de API de Plumber desplegada.
API_URL = "https://2ldfc4-massimo-barbetta.shinyapps.io/focus-ecg-api/prediccion_ecg"   
st.write("Herramienta de Diagnóstico Cardiovascular.")

# --- ESTRUCTURA PRINCIPAL DE PESTAÑAS ---

tab1, tab2, tab3 = st.tabs(["Análisis de Datos Reales", "Simulador de ECG", "Modelo de R (API)"])

# ---------------------------------------------
# PESTAÑA 1: CARGA Y ANÁLISIS DE DATOS REALES
# ---------------------------------------------
with tab1:
    st.header("Análisis de Datos Reales")
    st.write("Carga un archivo de ECG (.csv) para realizar un análisis detallado con tus modelos Keras.")
    
    # Aquí puedes añadir tu lógica de carga de modelos .keras
    # ... (Tu código para cargar best_model_mitbih.keras y best_model_ptbdb.keras)
    
    uploaded_file = st.file_uploader("Cargar Archivo de ECG (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_ecg = pd.read_csv(uploaded_file, header=None)
            st.success("Archivo cargado exitosamente. Listo para el preprocesamiento.")
            st.dataframe(df_ecg.head())
            
            # Aquí iría el código para preprocesar y predecir
            # if st.button("Analizar y Predecir"):
            #     st.info("Función de análisis en desarrollo...")
                
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}. Asegúrate de que es un archivo CSV válido.")


# ---------------------------------------------
# PESTAÑA 2: SIMULADOR DE SEÑALES DE ECG
# ---------------------------------------------
with tab2:
    st.header("Simulador de Señales de ECG")
    st.markdown("Ajusta el ritmo cardíaco para generar una señal ECG sintética en tiempo real.")
    
    # Widgets interactivos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bpm = st.slider("Ritmo Cardíaco (BPM)", 30, 150, 75, 5, key='bpm_sim')
        
    with col2:
        duration = st.slider("Duración (s)", 5, 20, 10, 1, key='duration_sim')

    with col3:
        # Control para el ruido
        noise_level = st.slider("Nivel de Ruido", 0.0, 0.5, 0.1, 0.05, key='noise_sim')

    # Generar la señal usando la función importada
    t, ecg_signal = generate_ecg(bpm, duration, noise_level)
    
    # Crear DataFrame para visualización con st.line_chart
    ecg_df = pd.DataFrame({'Tiempo (s)': t, 'Amplitud': ecg_signal}).set_index('Tiempo (s)')

    st.subheader(f"Señal Sintética Generada: {bpm} BPM")
    st.line_chart(ecg_df, use_container_width=True)
    
    st.info("Esta simulación utiliza funciones gaussianas para modelar las ondas P, QRS y T.")


# ---------------------------------------------
# PESTAÑA 3: PREDICCIÓN CON API DE R (PLUMBER)
# ---------------------------------------------
with tab3:
    st.header("Predicción con Modelo de R (API)")
    st.write("Introduce los parámetros del ECG para obtener la predicción del modelo desplegado en R/Plumber.")
    
    colA, colB = st.columns(2)

    with colA:
        frec_cardiaca = st.number_input(
            "Frecuencia Cardíaca (lpm):", 
            min_value=30.0, 
            max_value=200.0, 
            value=85.0, 
            step=0.1,
            key='frec_api'
        )

    with colB:
        var_rr = st.number_input(
            "Variabilidad RR (ms):", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.05, 
            step=0.001,
            key='var_api'
        )
    
    if st.button("Obtener Predicción de R", type="primary", key='btn_api_r'):
        datos_para_api = {
            "frecuencia_cardiaca": frec_cardiaca,
            "variabilidad_rr": var_rr
        }

        try:
            with st.spinner('Contactando con la API de R...'):
                respuesta = requests.post(API_URL, json=datos_para_api)
            
            if respuesta.status_code == 200:
                prediccion_json = respuesta.json()
                resultado = prediccion_json.get('prediccion', ['Resultado no encontrado'])[0]

                st.success("✅ Predicción Obtenida Exitosamente")
                st.metric(
                    label="Resultado del ECG", 
                    value=f"{resultado}",
                    delta="Modelo de R"
                )
                st.json(prediccion_json)

            else:
                st.error(f"❌ Error en la API: Código {respuesta.status_code}")
                st.code(respuesta.text, language='text')
                
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Error de Conexión: Asegúrate de que la API está funcionando y la URL es correcta ({API_URL}).")
            st.code(str(e), language='text')
            
    st.caption("Esta aplicación interactúa con un modelo de Machine Learning alojado como una API RESTful en R/Plumber.")
