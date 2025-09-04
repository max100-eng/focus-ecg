# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import cv2
from PIL import Image
import pandas as pd
import json
import os

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Focus ECG",
    page_icon="❤️",
    layout="wide"
)

# --- ESTILOS CSS ---
custom_theme_script = """
<style>
    body { background-color: #0E1117; color: #C8C9D0; }
    .stApp { background-color: #0E1117; }
    .stButton>button { background-color: #007BFF; color: white; border-radius: 5px; }
    .important-notice-box { background-color: #2F2F1C; border-left: 5px solid #FFD700; padding: 10px; border-radius: 5px; margin-top: 20px; }
    .st-emotion-cache-10q270i { background-color: #1A1A1A; border-radius: 8px; padding: 20px; }
    .st-emotion-cache-1n76qlr { background-color: #1A1A1A; }
    .red-border { border: 4px solid #FF4B4B; border-radius: 5px; padding: 5px; }
    .dataframe th, .dataframe td { background-color: #1A1A1A; color: #C8C9D0; }
    
    /* AÑADIDO: Font-face optimizado con font-display: swap; */
    @font-face {
      font-family: 'Source Sans Pro';
      src: url('ruta/a/la/fuente/SourceSansPro-Regular.woff2') format('woff2');
      font-weight: normal;
      font-style: normal;
      font-display: swap; 
    }
    
    @font-face {
      font-family: 'Source Sans Pro';
      src: url('ruta/a/la/fuente/SourceSansPro-SemiBold.woff2') format('woff2');
      font-weight: 600;
      font-style: normal;
      font-display: swap; 
    }
    
    @font-face {
      font-family: 'Source Sans Pro';
      src: url('ruta/a/la/fuente/SourceSansPro-Bold.woff2') format('woff2');
      font-weight: bold;
      font-style: normal;
      font-display: swap; 
    }
</style>
"""
st.markdown(custom_theme_script, unsafe_allow_html=True)

# Título de la aplicación
st.title("❤️ Focus ECG")
st.markdown("---")

## Funciones del modelo y preprocesamiento

@st.cache_resource
def load_models():
    """
    Carga ambos modelos (2D para imágenes y 1D para señales).
    """
    models = {}
    try:
        # Carga el modelo 2D para imágenes
        # Asegúrate de que este archivo contenga un modelo 2D.
        models['image_model'] = keras.models.load_model('best_model_2d.keras')
        st.info("✅ Modelo de imágenes (2D) cargado.")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo de imágenes: {e}")

    try:
        # Carga el modelo 1D para señales (Wavelet)
        # Asegúrate de que este archivo exista en la misma carpeta.
        models['signal_model'] = keras.models.load_model('best_model_wavelet.keras')
        st.info("✅ Modelo de señales (1D) cargado.")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo de señales: {e}")

    return models

def preprocess_image(image_bytes, img_size=(224, 224)):
    """
    Procesa y optimiza una imagen subida para un modelo 2D.
    """
    try:
        image = Image.open(image_bytes).convert('RGB')
        image_resized = image.resize(img_size)
        image_np = np.array(image_resized)
        image_normalized = image_np.astype('float32') / 255.0
        return np.expand_dims(image_normalized, axis=0)
    except Exception as e:
        st.error(f"❌ Error en el procesamiento de la imagen: {e}. Asegúrate de que la imagen sea un ECG claro.")
        return None

def preprocess_signal(file_bytes):
    """
    Procesa un archivo de señal (CSV) para un modelo 1D.
    """
    try:
        df = pd.read_csv(file_bytes)
        # Asume que la última columna es la etiqueta y la excluye
        signal = df.iloc[:, :-1].values.flatten()
        # Asegura que la longitud de la señal sea la esperada por el modelo (ej. 188 para PTBDB)
        if signal.shape[0] != 188:
            st.warning(f"La señal tiene una longitud de {signal.shape[0]}, pero el modelo espera 188.")
            return None
        
        signal_processed = signal.astype('float32')
        # Redimensiona para un modelo Conv1D
        return np.expand_dims(signal_processed, axis=-1)
    except Exception as e:
        st.error(f"❌ Error en el procesamiento de la señal: {e}. Asegúrate de que el archivo sea un CSV de señal de ECG.")
        return None

def find_last_conv_layer(model):
    """Encuentra la última capa convolucional 2D del modelo."""
    for layer in reversed(model.layers):
        if "Conv2D" in str(type(layer)):
            return layer
    return None

def generate_heatmap_2d(model, data_processed):
    """Genera un mapa de calor para un modelo 2D (Grad-CAM)."""
    try:
        last_conv_layer = find_last_conv_layer(model)
        
        if not last_conv_layer:
            st.warning("No se encontró una capa convolucional 2D para generar el heatmap.")
            return None
        
        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(data_processed)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_resized = cv2.resize(heatmap.numpy(), (224, 224))
        
        return heatmap_resized
    except Exception as e:
        st.error(f"❌ Error al generar el mapa de calor: {e}")
        return None

def interpret_model_output(prediction, model_type):
    """Interpreta la salida numérica del modelo."""
    if model_type == 'image':
        class_names = ["Ritmo sinusal normal", "Infarto Agudo del Miocardio (IAM)", "Arritmia", "Bloqueo de Branca"]
    else: # type == 'signal'
        class_names = ["Normal", "Infarto"] # Ejemplo de clases para señales
    
    predicted_class_index = np.argmax(prediction)
    diagnostico = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    
    reporte = {
        "Confianza del diagnóstico (%)": f"{confidence * 100:.2f}",
        "Observaciones": f"Predicción del modelo: {diagnostico}"
    }
    return {"diagnostico": diagnostico, "analisis_detallado": reporte}

## Diseño de la aplicación

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("Análisis de ECG")
    st.markdown("""
        <div class="important-notice-box">
        <h5 style="color: #FFD700; margin: 0;">AVISO IMPORTANTE:</h5>
        <p style="color: #FFD700; margin-top: 5px;">
        Este análisis es **solo para fines informativos y de demostración** y no constituye un diagnóstico médico.
        </p>
        </div>
    """, unsafe_allow_html=True)
    st.subheader("Subir ECG")
    
    uploaded_file = st.file_uploader("Sube un archivo ECG", type=['png', 'jpg', 'jpeg', 'csv'])
    url_input = st.text_input("...o introduce la URL de una imagen", help="Pega una URL y presiona Enter")
    analyze_button = st.button("Analizar")

    if 'processed' not in st.session_state:
        st.session_state['processed'] = False
        st.session_state['last_uploaded_file'] = None
        st.session_state['last_file_name'] = ""
        st.session_state['last_uploaded_file_type'] = None
        st.session_state['results'] = None

    if analyze_button:
        models = load_models()
        if not models:
            st.warning("No se pudieron cargar los modelos. No se puede continuar.")
            st.session_state['processed'] = False
        else:
            source_file, file_type, file_name = None, None, None
            if uploaded_file:
                source_file, file_type, file_name = uploaded_file, uploaded_file.type, uploaded_file.name
            elif url_input:
                try:
                    response = requests.get(url_input)
                    response.raise_for_status()
                    source_file = BytesIO(response.content)
                    source_file.seek(0)
                    file_type = 'image/png' if 'png' in url_input.lower() else 'image/jpeg'
                    file_name = url_input
                    st.success("Imagen de URL cargada exitosamente!")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Error al descargar la imagen de la URL: {e}")
                    source_file = None

            if source_file is not None:
                st.session_state.update({
                    'last_uploaded_file': source_file,
                    'last_uploaded_file_type': file_type,
                    'last_file_name': file_name,
                    'results': None,
                    'processed': False
                })

                with st.spinner("Procesando señal ECG..."):
                    if 'image' in file_type:
                        if 'image_model' in models:
                            data_for_prediction = preprocess_image(source_file)
                            if data_for_prediction is not None:
                                prediction = models['image_model'].predict(data_for_prediction)
                                heatmap_data = generate_heatmap_2d(models['image_model'], data_for_prediction)
                                results = interpret_model_output(prediction, 'image')
                                results['heatmap_data'] = heatmap_data
                            else:
                                results = None
                        else:
                            st.error("No se pudo cargar el modelo de imágenes.")
                            results = None
                    elif file_type == 'text/csv':
                        if 'signal_model' in models:
                            data_for_prediction = preprocess_signal(source_file)
                            if data_for_prediction is not None:
                                data_for_prediction = data_for_prediction.reshape(1, -1, 1) 
                                prediction = models['signal_model'].predict(data_for_prediction)
                                results = interpret_model_output(prediction, 'signal')
                            else:
                                results = None
                        else:
                            st.error("No se pudo cargar el modelo de señales.")
                            results = None
                    else:
                        st.warning("Formato de archivo no soportado.")
                        results = None

                    if results:
                        st.session_state.update({'results': results, 'processed': True})
                        st.success("Procesamiento completado!")
                    else:
                        st.session_state['processed'] = False

with col2:
    if 'last_uploaded_file' in st.session_state and st.session_state['last_uploaded_file'] is not None:
        st.subheader("ECG Subido")
        st.session_state['last_uploaded_file'].seek(0)
        st.image(st.session_state['last_uploaded_file'], caption=st.session_state.get('last_file_name', 'ECG'))
        st.markdown("---")
    
    if 'processed' in st.session_state and st.session_state['processed']:
        st.subheader("Resultados del análisis:")
        results = st.session_state['results']
        
        if 'heatmap_data' in results and results.get('heatmap_data') is not None:
            st.subheader("ECG Subido con Heatmap")
            uploaded_image_bytes = st.session_state['last_uploaded_file']
            uploaded_image_bytes.seek(0)
            original_image = Image.open(uploaded_image_bytes).convert('RGB')
            original_image_np = np.array(original_image)
            heatmap_data = results['heatmap_data']
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(original_image_np, aspect='auto')
            heatmap_resized_for_display = cv2.resize(heatmap_data, (original_image_np.shape[1], original_image_np.shape[0]))
            ax.imshow(heatmap_resized_for_display, cmap='hot', alpha=0.5)
            ax.set_axis_off()
            st.pyplot(fig)
        else:
            st.warning("No se pudo generar el heatmap.")

        st.subheader("Diagnóstico")
        diagnostico = results['diagnostico']
        
        if "infarto" in diagnostico.lower():
            st.error(f"⚠️ **DIAGNÓSTICO: {diagnostico}**")
        elif "normal" in diagnostico.lower():
            st.success(f"✅ {diagnostico}")
        else:
            st.warning(f"⚠️ {diagnostico}")
            
        st.subheader("Análisis Detallado de Elementos del ECG")
        analisis_df = pd.DataFrame(results['analisis_detallado'].items(), columns=['Elemento', 'Estado'])
        st.table(analisis_df)

        st.markdown("---")
        st.subheader("Descargar Reporte")
        reporte_json = json.dumps(results['analisis_detallado'], indent=4)
        st.download_button(
            label="Descargar Informe de Análisis (.json)",
            data=reporte_json,
            file_name="reporte_ecg.json",
            mime="application/json"
        )
    else:
        st.subheader("Resultados del análisis:")
        st.warning("Por favor, sube y procesa un archivo ECG para ver el informe.")