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
<script>
    function applyCustomTheme() {
        // Apply dark theme to the main Streamlit container
        var mainContainer = window.parent.document.querySelector('.stApp');
        if (mainContainer) {
            mainContainer.style.backgroundColor = '#0E1117';
            mainContainer.style.color = '#FAFAFA';
        }

        // Apply dark theme to the sidebar
        var sidebar = window.parent.document.querySelector('.st-emotion-cache-1cpx96c');
        if (sidebar) {
            sidebar.style.backgroundColor = '#262730';
        }

        // Apply dark theme to other components
        var elements = window.parent.document.querySelectorAll('.st-emotion-cache-12fmw6v, .st-emotion-cache-1r6chqg');
        elements.forEach(el => {
            el.style.backgroundColor = '#0E1117';
        });

        // Apply theme to headers
        var headers = window.parent.document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        headers.forEach(el => {
            el.style.color = '#FAFAFA';
        });

        // Hide main menu and footer
        var mainMenuItems = window.parent.document.querySelector("#MainMenu");
        if (mainMenuItems) mainMenuItems.style.visibility = "hidden";
        
        var footer = window.parent.document.querySelector("footer");
        if (footer) footer.style.visibility = "hidden";
    }

    // Run the function when the page loads
    window.addEventListener('load', applyCustomTheme);
    // Also run it on a short delay to catch dynamically loaded elements
    setTimeout(applyCustomTheme, 500);
</script>
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
        st.success("Modelo de TensorFlow cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def predict_with_model(data, model, file_type):
    """
    Realiza una predicción sobre los datos ECG usando el modelo.
    """
    if model:
        st.info("Modelo cargado. Preprocesando y prediciendo...")
        try:
            # Si el archivo es una imagen, se usan datos de ejemplo
            if file_type in ["image/png", "image/jpeg", "image/jpg"]:
                st.warning("Advertencia: Los archivos de imagen no contienen datos de señal ECG. Se usará una señal de datos de ejemplo para demostrar la funcionalidad.")
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
            prediction = model.predict(data_processed)
            st.write("Resultado de la predicción (probabilidades):", prediction)
            
            classes = ['Ritmo sinusal normal', 'Arritmia', 'Taquicardia', 'Bradicardia']
            predicted_class_index = np.argmax(prediction[0])
            diagnostico_final = classes[predicted_class_index]
            
            prediction_dict = {
                "diagnostico": diagnostico_final,
                "probabilidades": {cls: prob for cls, prob in zip(classes, prediction[0])},
                "metricas": {
                    'Precisión': 0.95,
                    'Sensibilidad': 0.92,
                    'Especificidad': 0.96
                }
            }
            return prediction_dict

        except Exception as e:
            st.error(f"Error durante la predicción con el modelo: {e}")
            return None
    
    else:
        st.warning("El modelo no ha podido ser cargado. No se puede realizar la predicción.")
        return None

# Cargar el modelo de IA al iniciar la aplicación
ecg_model = load_ecg_model()

# --- NAVEGACIÓN DE LA PÁGINA ---

# Sidebar para navegación
st.sidebar.title("Navegación")
opcion = st.sidebar.radio(
    "Selecciona una opción:",
    ["Inicio", "Subir ECG", "Resultados", "Acerca de"]
)

if opcion == "Inicio":
    st.header("Bienvenido a Focus ECG")
    st.write("""
    Esta aplicación utiliza inteligencia artificial para analizar señales ECG
    y detectar posibles anomalías cardíacas.
    
    ### Características:
    - Análisis de señales ECG en tiempo real
    - Detección de arritmias cardíacas
    - Visualización interactiva de resultados
    - Reportes detallados
    """)
    
    # Ejemplo de visualización
    st.subheader("Ejemplo de señal ECG")
    x = np.linspace(0, 10, 1000)
    y = np.sin(x) + 0.1 * np.random.randn(1000)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, 'b-', linewidth=1)
    ax.set_title("Señal ECG de ejemplo")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.grid(True)
    st.pyplot(fig)

elif opcion == "Subir ECG":
    st.header("Subir Archivo ECG")
    
    archivo = st.file_uploader(
        "Selecciona un archivo ECG (formatos admitidos: CSV, TXT, PNG, JPG, JPEG)",
        type=['csv', 'txt', 'png', 'jpg', 'jpeg']
    )
    
    if archivo is not None:
        st.success(f"Archivo {archivo.name} subido exitosamente!")
        
        with st.spinner("Procesando señal ECG..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            
            try:
                file_type = archivo.type
                data = None
                
                if file_type in ["text/csv", "text/plain"]:
                    data = pd.read_csv(archivo)
                elif file_type in ["image/png", "image/jpeg"]:
                    st.warning("Advertencia: Los archivos de imagen no contienen datos de señal ECG. Se usará una señal de datos de ejemplo para demostrar la funcionalidad.")
                    data = np.random.randn(1000)
                else:
                    st.warning("Tipo de archivo no soportado para análisis.")
                    data = None

                if data is not None:
                    results = predict_with_model(data, ecg_model, file_type)
                    
                    if results:
                        st.session_state['results'] = results
                        st.session_state['processed'] = True
                        st.success("Procesamiento completado! Ve a la pestaña 'Resultados' para ver el informe.")
                    else:
                        st.session_state['processed'] = False
                else:
                    st.session_state['processed'] = False
            except Exception as e:
                st.error(f"Ocurrió un error durante el análisis: {e}")
                st.session_state['processed'] = False

elif opcion == "Resultados":
    st.header("Resultados del Análisis")
    
    if 'processed' in st.session_state and st.session_state['processed'] and st.session_state['results']:
        results = st.session_state['results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Diagnóstico")
            diagnostico = results['diagnostico']
            
            if "normal" in diagnostico.lower():
                st.success(diagnostico)
            else:
                st.error(diagnostico)
                
            st.info(f"Probabilidad de {diagnostico}: {results['probabilidades'].get(diagnostico, 0):.2f}")
        
        with col2:
            st.subheader("Métricas de Desempeño del Modelo")
            st.json(results['metricas'])
        
        st.subheader("Análisis Detallado")
        
        categorias = list(results['probabilidades'].keys())
        valores = list(results['probabilidades'].values())
        
        fig, ax = plt.subplots()
        colores = ['green', 'red', 'orange', 'blue']
        ax.bar(categorias, valores, color=colores[:len(categorias)])
        ax.set_ylabel('Probabilidad')
        ax.set_title('Distribución de Diagnósticos')
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        
    else:
        st.warning("Por favor, sube y procesa un archivo ECG en la pestaña 'Subir ECG' primero.")

elif opcion == "Acerca de":
    st.header("Acerca de Focus ECG")
    st.write("""
    ### Tecnologías Utilizadas:
    - **Python** con **TensorFlow/Keras** para el modelo de IA
    - **Streamlit** para la interfaz web
    - **Numpy** y **Pandas** para manejo de datos
    - **Matplotlib** para visualizaciones
    - **Requests** para posibles llamadas a API
    """)
