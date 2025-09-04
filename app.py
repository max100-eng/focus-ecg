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

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Focus ECG",
    page_icon="❤️",
    layout="wide"
)

# --- ESTILOS CSS ---
@font-face {
  font-family: 'Source Sans Pro';
  src: url('ruta/a/la/fuente/SourceSansPro-Regular.woff2') format('woff2');
  font-weight: normal;
  font-style: normal;
  font-display: swap; /* Esto es lo que necesitas agregar */
}

@font-face {
  font-family: 'Source Sans Pro';
  src: url('ruta/a/la/fuente/SourceSansPro-SemiBold.woff2') format('woff2');
  font-weight: 600;
  font-style: normal;
  font-display: swap; /* Y aquí también */
}

@font-face {
  font-family: 'Source Sans Pro';
  src: url('ruta/a/la/fuente/SourceSansPro-Bold.woff2') format('woff2');
  font-weight: bold;
  font-style: normal;
  font-display: swap; /* Y en todas tus fuentes web */
}

# Título de la aplicación
st.title("❤️ Focus ECG")
st.markdown("---")

## Funciones del modelo y preprocesamiento

@st.cache_resource
def load_ecg_2d_model():
    """
    Carga el modelo 2D de ECG.
    ⚠️ IMPORTANTE: Asegúrate de que tu modelo 'modelo_ecg_2d.h5' esté en la misma carpeta.
    """
    try:
        model = keras.models.load_model('modelo_ecg_2d.h5')
        st.info("✅ Modelo 2D cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}. Asegúrate de que 'modelo_ecg_2d.h5' esté en la misma carpeta.")
        return None

def find_last_conv_layer(model):
    """Encuentra la última capa convolucional 2D del modelo."""
    for layer in reversed(model.layers):
        if "Conv2D" in str(type(layer)):
            return layer
    return None

def interpret_model_output(prediction):
    """Interpreta la salida numérica del modelo."""
    class_names = ["Ritmo sinusal normal", "Infarto Agudo del Miocardio (IAM)", "Arritmia", "Bloqueo de Branca"]
    predicted_class_index = np.argmax(prediction)
    diagnostico = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    
    reporte = {
        "Confianza del diagnóstico (%)": f"{confidence * 100:.2f}",
        "Observaciones": f"Predicción del modelo: {diagnostico}"
    }
    return {"diagnostico": diagnostico, "analisis_detallado": reporte}

def process_uploaded_image_for_2d_model(image_bytes, img_size=(224, 224)):
    """Procesa una imagen subida para que sea compatible con un modelo 2D."""
    try:
        image = Image.open(image_bytes).convert('RGB')
        image_resized = image.resize(img_size)
        image_np = np.array(image_resized)
        image_normalized = image_np.astype('float32') / 255.0
        return image_normalized
    except Exception as e:
        st.error(f"❌ Error en el procesamiento de la imagen: {e}. Asegúrate de que la imagen sea un ECG claro.")
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

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10) # Añadido 1e-10 para evitar división por cero
        heatmap_resized = cv2.resize(heatmap.numpy(), (224, 224))
        
        return heatmap_resized
    except Exception as e:
        st.error(f"❌ Error al generar el mapa de calor: {e}")
        return None

def predict_with_2d_model(data, file_type):
    """Función principal que realiza la predicción con el modelo 2D."""
    ecg_model = load_ecg_2d_model()
    if ecg_model is None:
        return None

    try:
        image_processed = process_uploaded_image_for_2d_model(data)
        if image_processed is None:
            return None
        
        data_for_prediction = np.expand_dims(image_processed, axis=0)
        
        st.info("Modelo cargado. Preprocesando y prediciendo...")
        
        prediction = ecg_model.predict(data_for_prediction)

        heatmap_data = generate_heatmap_2d(ecg_model, data_for_prediction)

        results = interpret_model_output(prediction)
        results["heatmap_data"] = heatmap_data
        
        return results

    except Exception as e:
        st.error(f"❌ Error durante la predicción con el modelo: {e}")
        return None

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
    
    uploaded_file = st.file_uploader("Sube un archivo ECG", type=['png', 'jpg', 'jpeg'])
    url_input = st.text_input("...o introduce la URL de una imagen", help="Pega una URL y presiona Enter")
    analyze_button = st.button("Analizar")

    if 'processed' not in st.session_state:
        st.session_state['processed'] = False
        st.session_state['last_uploaded_file'] = None
        st.session_state['last_file_name'] = ""
        st.session_state['last_uploaded_file_type'] = None
        st.session_state['results'] = None

    if analyze_button:
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
                results = predict_with_2d_model(source_file, file_type)
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

        if results and results.get('heatmap_data') is not None:
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
        
        if diagnostico == "Infarto Agudo del Miocardio (IAM)":
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