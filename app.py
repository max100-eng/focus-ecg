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

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Focus ECG",
    page_icon="❤️",
    layout="wide"
)

# --- INICIO: ESTILOS CSS ---
custom_theme_script = """
<style>
    body { background-color: #0E1117; color: #C8C9D0; }
    .stApp { background-color: #0E1117; }
    .stButton>button { background-color: #007BFF; color: white; border-radius: 5px; }
    .important-notice-box { background-color: #2F2F1C; border-left: 5px solid #FFD700; padding: 10px; border-radius: 5px; margin-top: 20px; }
</style>
"""
st.markdown(custom_theme_script, unsafe_allow_html=True)
# --- FIN: ESTILOS CSS ---

# Título de la aplicación
st.title("❤️ Focus ECG")
st.markdown("---")

# --- FUNCIONES CLAVE DEL MODELO Y PROCESAMIENTO ---

@st.cache_resource
def load_ecg_transfer_model():
    """Carga el modelo de aprendizaje por transferencia."""
    try:
        model = keras.models.load_model('transfer_model.h5')
        st.info("✅ Modelo de aprendizaje por transferencia cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}. Asegúrate de que 'transfer_model.h5' esté en la misma carpeta.")
        return None

def find_last_conv_layer(model):
    """Encuentra la última capa convolucional 2D."""
    for layer in reversed(model.layers):
        if 'conv' in layer.name: # Generalizado para capas convolucionales 2D
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

def create_ecg_image_from_signal(signal_1d, img_size=(224, 224)):
    """Convierte una señal de ECG 1D en una imagen 2D para modelos de visión."""
    img = np.zeros(img_size, dtype=np.uint8)
    scaled_signal = (signal_1d - np.min(signal_1d)) / (np.max(signal_1d) - np.min(signal_1d))
    scaled_signal = (scaled_signal * (img_size[0] - 1)).astype(int)
    
    for i, y in enumerate(scaled_signal):
        if i < img_size[1]:
            img[y, i] = 255
            
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img_rgb

def process_uploaded_image(image_bytes):
    """Procesa una imagen subida para extraer la señal 1D."""
    try:
        image = Image.open(image_bytes).convert('RGB')
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        signal = [np.argmin(gray_image[:, col]) for col in range(gray_image.shape[1])]
        signal_array = np.array(signal, dtype=np.float32)

        if len(signal_array) > 1000:
            signal_array = signal_array[np.linspace(0, len(signal_array) - 1, 1000).astype(int)]
        elif len(signal_array) < 1000:
            padding = np.zeros(1000 - len(signal_array))
            signal_array = np.concatenate((signal_array, padding))
        
        return signal_array.reshape(1000, 1)

    except Exception as e:
        st.error(f"❌ Error en el procesamiento de la imagen: {e}. Asegúrate de que la imagen sea un ECG claro.")
        return None

def predict_with_transfer_model(data, file_type):
    """Función principal que realiza la predicción con el modelo de transferencia."""
    ecg_model = load_ecg_transfer_model()
    if not ecg_model:
        return None

    try:
        # Paso 1: Obtener la señal 1D de la imagen subida
        signal_1d = process_uploaded_image(data)
        if signal_1d is None:
            return None

        # Paso 2: Convertir la señal 1D a una imagen 2D para el modelo
        ecg_image = create_ecg_image_from_signal(signal_1d.flatten())
        data_processed = np.expand_dims(ecg_image, axis=0)
        
        # Paso 3: Realizar la predicción
        st.info("Modelo cargado. Preprocesando y prediciendo...")
        prediction = ecg_model.predict(data_processed)

        # Paso 4: Generar el mapa de calor (Grad-CAM)
        heatmap_data = generate_heatmap_2d(ecg_model, data_processed)

        # Paso 5: Interpretar y devolver los resultados
        results = interpret_model_output(prediction)
        results["heatmap_data"] = heatmap_data
        
        return results

    except Exception as e:
        st.error(f"❌ Error durante la predicción con el modelo: {e}")
        return None

def generate_heatmap_2d(model, data_processed):
    """Genera un mapa de calor para un modelo 2D (Grad-CAM)."""
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

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = tf.image.resize(tf.expand_dims(heatmap, axis=-1), (224, 224))
    
    return heatmap.numpy().squeeze()

# --- DISEÑO DE LA APLICACIÓN DE UNA SOLA PÁGINA ---

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
                results = predict_with_transfer_model(source_file, file_type)
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

        if 'last_uploaded_file_type' in st.session_state and \
           st.session_state['last_uploaded_file_type'] in ["image/png", "image/jpeg", "image/jpg", "image/unknown_url_image"]:
            
            st.subheader("ECG Subido con Heatmap")
            
            uploaded_image_bytes = st.session_state['last_uploaded_file']
            uploaded_image_bytes.seek(0)
            original_image = Image.open(uploaded_image_bytes).convert('RGB')
            original_image_np = np.array(original_image)
            
            heatmap_data = results['heatmap_data']
            if heatmap_data is not None:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(original_image_np, aspect='auto')
                ax.imshow(heatmap_data, cmap='hot', alpha=0.5, extent=[0, original_image_np.shape[1], original_image_np.shape[0], 0])
                ax.set_axis_off()
                st.pyplot(fig)
            else:
                st.warning("No se pudo generar el heatmap.")

        st.subheader("Diagnóstico")
        diagnostico = results['diagnostico']
        
        if diagnostico == "Infarto Agudo del Miocardio (IAM)":
            st.error(f"⚠️ **DIAGNÓSTICO: {diagnostico}**")
            st.warning("Busque **ATENCIÓN MÉDICA DE URGENCIA** de inmediato.")
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
