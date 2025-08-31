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
import cv2
from PIL import Image

# Streamlit page configuration (title, layout, and custom theme)
st.set_page_config(
    page_title="Focus ECG",
    page_icon="❤️",
    layout="wide"
)

# --- INICIO: CÓDIGO CSS MEJORADO PARA LEGIBILIDAD ---
custom_theme_script = """
<style>
    /* Estilos generales del tema oscuro */
    body {
        background-color: #0E1117; /* Fondo principal oscuro */
        color: #C8C9D0; /* Texto claro pero con buen contraste */
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }
    .stApp {
        background-color: #0E1117;
    }
    .st-emotion-cache-1cpx96v { /* Sidebar */
        background-color: #1F2228;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF; /* Títulos en blanco puro para que resalten */
    }
    .stButton>button {
        background-color: #007BFF; /* Azul de botón más estándar y visible */
        color: white;
        border-radius: 5px;
    }
    .st-emotion-cache-12fmw6v, .st-emotion-cache-1r6chqg { /* Contenedores principales */
        background-color: #0E1117;
    }
    /* Ocultar el menú y el pie de página de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Estilo para el aviso importante, mejorando el contraste */
    .important-notice-box {
        background-color: #2F2F1C;
        border-left: 5px solid #FFD700;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .important-notice-box h5, .important-notice-box p {
        color: #FFD700;
    }
    
    /* Estilo para el recuadro de resultados */
    .st-emotion-cache-10q270i {
        background-color: #1A1A1A;
        border-radius: 8px;
        padding: 20px;
    }
    .st-emotion-cache-1n76qlr {
        background-color: #1A1A1A;
    }
    
    /* Estilo del recuadro con borde rojo para la imagen */
    .red-border {
        border: 4px solid #FF4B4B;
        border-radius: 5px;
        padding: 5px;
    }
    
    /* Mejorar la legibilidad de la tabla */
    .dataframe th, .dataframe td {
        background-color: #1A1A1A;
        color: #C8C9D0;
    }
    
</style>
"""

st.markdown(custom_theme_script, unsafe_allow_html=True)
# --- FIN: CÓDIGO CSS MEJORADO PARA LEGIBILIDAD ---

# Título de la aplicación
st.title("❤️ Focus ECG")
st.markdown("---")

# --- FUNCIONES DE ANÁLISIS ---

@st.cache_resource
def load_ecg_model():
    """
    Carga el modelo de IA una sola vez y lo retorna.
    """
    try:
        model = keras.models.load_model('modelo_ecg.h5')
        st.info("Modelo de TensorFlow cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}. Asegúrate de que 'modelo_ecg.h5' esté en la misma carpeta y sea accesible.")
        return None

def find_last_conv_layer(model):
    """Encuentra la última capa convolucional 1D en el modelo."""
    for layer in reversed(model.layers):
        if 'conv1d' in layer.name:
            return layer
    return None

def interpret_model_output(prediction):
    """
    Interpreta la salida numérica del modelo y la convierte en un diagnóstico.
    """
    # Define tus nombres de clase aquí. Deben coincidir con el orden de las etiquetas
    # que usaste para entrenar el modelo.
    class_names = ["Ritmo sinusal normal", "Infarto Agudo del Miocardio (IAM)", "Arritmia", "Bloqueo de Branca"]
    
    predicted_class_index = np.argmax(prediction)
    diagnostico = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    
    reporte = {
        "Confianza del diagnóstico (%)": f"{confidence * 100:.2f}",
        "Observaciones": f"Predicción del modelo: {diagnostico}"
    }
    
    return {"diagnostico": diagnostico, "analisis_detallado": reporte}

def generate_heatmap(model, data_processed):
    """
    Genera un mapa de calor real usando la técnica de Grad-CAM.
    """
    last_conv_layer = find_last_conv_layer(model)
    if not last_conv_layer:
        st.warning("No se encontró una capa Conv1D para generar el heatmap.")
        return np.zeros(data_processed.shape[1])
        
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(data_processed)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    pooled_grads = tf.reduce_mean(grads, axis=0)
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def process_ecg_image(image_bytes):
    """
    Lee una imagen de ECG, la convierte en una señal numérica, la redimensiona y la normaliza.
    """
    try:
        # Paso 1: Leer la imagen desde los bytes
        image = Image.open(image_bytes).convert('RGB')
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Paso 2: Extraer la señal de la línea más oscura
        # Encuentra los píxeles más oscuros por columna, lo que representa la señal
        signal = []
        for col in range(gray_image.shape[1]):
            # Encuentra el índice de la fila con el valor de píxel mínimo (más oscuro)
            row_index = np.argmin(gray_image[:, col])
            signal.append(row_index)

        signal_array = np.array(signal, dtype=np.float32)

        # Paso 3: Asegurar que la señal tenga la longitud de 1000
        if len(signal_array) > 1000:
            signal_array = signal_array[np.linspace(0, len(signal_array) - 1, 1000).astype(int)]
        elif len(signal_array) < 1000:
            padding = np.zeros(1000 - len(signal_array))
            signal_array = np.concatenate((signal_array, padding))
        
        # Paso 4: Normalizar los datos
        signal_array = (signal_array - np.min(signal_array)) / (np.max(signal_array) - np.min(signal_array))
        
        # El modelo espera una forma de (1000, 1), por lo que necesitamos un reshape final
        return signal_array.reshape(1000, 1)

    except Exception as e:
        st.error(f"Error en el procesamiento de la imagen: {e}. Asegúrate de que la imagen sea un ECG claro.")
        return None

def predict_with_model(data, file_type):
    """
    Realiza una predicción sobre los datos ECG usando el modelo.
    """
    ecg_model = load_ecg_model()
    if not ecg_model:
        st.warning("El modelo no ha podido ser cargado. No se puede realizar la predicción.")
        return None

    st.info("Modelo cargado. Preprocesando y prediciendo...")
    
    try:
        if file_type in ["image/png", "image/jpeg", "image/jpg", "image/unknown_url_image"]:
            data_processed = process_ecg_image(data)
            if data_processed is None:
                return None
        else:
            # Lógica para otros tipos de archivo, si es necesario.
            st.error("Tipo de archivo no soportado para este análisis.")
            return None

        # --- LÓGICA CORREGIDA ---
        # Primero, realiza la predicción para "llamar" a las capas del modelo
        prediction = ecg_model.predict(data_processed[np.newaxis, ...])
        
        # Luego, genera el mapa de calor con el modelo ya "construido"
        heatmap_data = generate_heatmap(ecg_model, data_processed[np.newaxis, ...])
        
        results = interpret_model_output(prediction)
        results["heatmap_data"] = heatmap_data
        
        return results

    except Exception as e:
        st.error(f"Error durante la predicción con el modelo: {e}")
        return None
        

# --- DISEÑO DE LA APLICACIÓN DE UNA SOLA PÁGINA ---

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("Análisis de ECG")
    st.write("Sube una imagen o usa la URL de un electrocardiograma.")
    st.write("La IA te proporcionará un resumen detallado, las mediciones principales y un mapa de calor (heatmap).")
    
    st.markdown("""
        <div class="important-notice-box">
        <h5 style="color: #FFD700; margin: 0;">AVISO IMPORTANTE:</h5>
        <p style="color: #FFD700; margin-top: 5px;">
        Este análisis es **solo para fines informativos y de demostración** y no constituye un diagnóstico médico.
        Siempre consulta a un profesional de la salud calificado para una interpretación precisa
        de cualquier dato médico.
        </p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Subir ECG")
    
    uploaded_file = st.file_uploader(
        "Sube un archivo ECG",
        type=['csv', 'txt', 'png', 'jpg', 'jpeg']
    )
    
    url_input = st.text_input("...o introduce la URL de una imagen", help="Pega una URL y presiona Enter")
    
    analyze_button = st.button("Analizar")

    # Mantiene el estado del archivo subido en la sesión
    if 'processed' not in st.session_state:
        st.session_state['processed'] = False
        st.session_state['last_uploaded_file'] = None
        st.session_state['last_file_name'] = ""
        st.session_state['last_uploaded_file_type'] = None
        st.session_state['results'] = None

    if analyze_button:
        source_file = None
        file_type = None
        file_name = None
        
        if uploaded_file:
            source_file = uploaded_file
            file_type = uploaded_file.type
            file_name = uploaded_file.name
        elif url_input:
            try:
                response = requests.get(url_input)
                response.raise_for_status()
                source_file = BytesIO(response.content)
                source_file.seek(0)
                if 'png' in url_input.lower():
                    file_type = 'image/png'
                elif 'jpg' in url_input.lower() or 'jpeg' in url_input.lower():
                    file_type = 'image/jpeg'
                else:
                    file_type = 'image/unknown_url_image'
                file_name = url_input
                st.success("Imagen de URL cargada exitosamente!")
            except requests.exceptions.RequestException as e:
                st.error(f"Error al descargar la imagen de la URL: {e}")
                source_file = None
        
        if source_file is not None:
            st.session_state['last_uploaded_file'] = source_file
            st.session_state['last_uploaded_file_type'] = file_type
            st.session_state['last_file_name'] = file_name
            st.session_state['results'] = None
            st.session_state['processed'] = False

            with st.spinner("Procesando señal ECG..."):
                results = predict_with_model(source_file, file_type)
                if results:
                    st.session_state['results'] = results
                    st.session_state['processed'] = True
                    st.success("Procesamiento completado!")
                else:
                    st.session_state['processed'] = False

with col2:
    if 'last_uploaded_file' in st.session_state and st.session_state['last_uploaded_file'] is not None:
        st.subheader("ECG Subido")
        # Asegurarse de que el puntero del archivo esté al inicio para poder leerlo
        st.session_state['last_uploaded_file'].seek(0)
        st.image(st.session_state['last_uploaded_file'], caption=st.session_state.get('last_file_name', 'ECG'))
        st.markdown("---")
    
    if 'processed' in st.session_state and st.session_state['processed']:
        st.subheader("Resultados del análisis:")
        results = st.session_state['results']

        if 'last_uploaded_file_type' in st.session_state and \
           st.session_state['last_uploaded_file_type'] in ["image/png", "image/jpeg", "image/jpg", "image/unknown_url_image"]:
            
            st.subheader("ECG Subido con Heatmap")
            
            uploaded_image_bytes = st.session_state['last_uploaded_file']
            uploaded_image_bytes.seek(0)
            original_image = Image.open(uploaded_image_bytes).convert('RGB')
            original_image_np = np.array(original_image)
            
            heatmap_data = results['heatmap_data']
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(original_image_np, aspect='auto')
            
            # Ajustar la longitud del heatmap para que coincida con el ancho de la imagen
            heatmap_display = np.interp(np.linspace(0, 1, original_image_np.shape[1]), 
                                        np.linspace(0, 1, len(heatmap_data)), 
                                        heatmap_data)
            
            cmap = plt.cm.get_cmap('hot')
            
            # Crear una máscara de calor para superponer sobre la imagen
            heatmap_mask = np.zeros_like(original_image_np[:,:,0], dtype=float)
            center_row = original_image_np.shape[0] // 2
            # Superponer el heatmap en el centro de la imagen
            heatmap_mask[center_row-10:center_row+10, :] = np.tile(heatmap_display, (20, 1))
            
            ax.imshow(heatmap_mask, cmap=cmap, alpha=0.5, extent=[0, original_image_np.shape[1], original_image_np.shape[0], 0])
            
            ax.set_axis_off()
            st.pyplot(fig)
            
        st.subheader("Diagnóstico")
        diagnostico = results['diagnostico']
        
        if diagnostico == "Infarto Agudo del Miocardio (IAM)":
            st.error(f"⚠️ **DIAGNÓSTICO: {diagnostico}**")
            st.warning("Busque **ATENCIÓN MÉDICA DE URGENCIA** de inmediato. Este resultado sugiere un posible evento cardíaco grave.")
        elif "normal" in diagnostico.lower():
            st.success(f"✅ {diagnostico}")
        else:
            st.warning(f"⚠️ {diagnostico}")
            
        st.subheader("Análisis Detallado de Elementos del ECG")
        analisis_df = pd.DataFrame(results['analisis_detallado'].items(), columns=['Elemento', 'Estado'])
        st.table(analisis_df)

    else:
        st.subheader("Resultados del análisis:")
        st.warning("Por favor, sube y procesa un archivo ECG para ver el informe.")
