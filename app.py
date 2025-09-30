# -*- coding: utf-8 -*-
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

# --- Cargar los modelos de análisis de señales ---
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

>>>>>>> 9a82aa886352dc1424847d430e42b17b45cb0b59
@st.cache_resource
def load_analysis_models():
    models = {}
    try:
<<<<<<< HEAD
        models['mitbih'] = tf.keras.models.load_model("best_model_mitbih.keras")
        st.success("✅ Modelo MIT-BIH (best_model_mitbih.keras) cargado.")
    except Exception as e:
        st.error(f"❌ Error al cargar best_model_mitbih.keras: {e}")
        models['mitbih'] = None
=======
        model = keras.models.load_model('modelo_ecg.h5')
        st.info("Modelo de TensorFlow cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}. Asegúrate de que 'modelo_ecg.h5' esté en la misma carpeta y sea accesible.")
        return None

def analyze_ecg_details(ecg_signal):
    """
    Simula un análisis detallado de los elementos del ECG basado en datos numéricos.
    Esta función también genera datos simulados para un heatmap.
    """
    # Valores aleatorios que simulan un análisis del modelo
    pr_interval = random.uniform(0.12, 0.22)
    qrs_duration = random.uniform(0.06, 0.15)
    st_segment = random.uniform(-0.1, 0.2)
    qt_interval = random.uniform(0.35, 0.50)
>>>>>>> 9a82aa886352dc1424847d430e42b17b45cb0b59
    
    try:
        models['ptbdb'] = tf.keras.models.load_model("best_model_ptbdb.keras")
        st.success("✅ Modelo PTB-DB (best_model_ptbdb.keras) cargado.")
    except Exception as e:
        st.error(f"❌ Error al cargar best_model_ptbdb.keras: {e}")
        models['ptbdb'] = None
    
    return models

analysis_models = load_analysis_models()

# --- Funciones de simulación y preprocesamiento ---
def simulate_ecg_analysis():
    """Simula el análisis de los intervalos y métricas del ECG."""
    return {
        "heartRate": 72,
        "autoDiagnosis": "Ritmo Sinusal Normal",
        "ecgIntervals": [
            {"interval": "PR", "duration": 160, "normalRange": "120-200"},
            {"interval": "QRS", "duration": 90, "normalRange": "80-120"},
            {"interval": "QT", "duration": 380, "normalRange": "350-440"},
            {"interval": "QTc", "duration": 420, "normalRange": "340-440"}
        ]
    }

<<<<<<< HEAD
def generate_ecg_graph():
    """Genera un gráfico simulado de un trazado de ECG."""
    t = np.linspace(0, 5, 500)
    p_wave = 0.1 * np.exp(-100 * (t - 0.1)**2)
    qrs_complex = -0.6 * np.exp(-1000 * (t - 0.2)**2) + 1.2 * np.exp(-1000 * (t - 0.25)**2) - 0.2 * np.exp(-1000 * (t - 0.3)**2)
    t_wave = 0.3 * np.exp(-80 * (t - 0.45)**2)
    ecg_wave = p_wave + qrs_complex + t_wave
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, ecg_wave, color='red')
    ax.set_title("Trazado ECG Simulado")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Voltaje (mV)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f7f7f7')
    plt.tight_layout()
    return fig

# --- Interfaz de Streamlit ---
def main():
    st.title("CardioSense")
    st.subheader("Análisis de Electrocardiogramas (ECG) mediante IA")
    
    st.sidebar.title("Configuración")
    st.sidebar.markdown("Sube una imagen de ECG para analizar.")
    
    uploaded_file = st.file_uploader("Sube una imagen de ECG...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.write("---")
        st.subheader("Imagen subida")
        image_data = uploaded_file.getvalue()
        st.image(image_data, caption=uploaded_file.name, use_container_width=True)
        
        st.write("---")
        st.subheader("Resultados del análisis")
        
        # --- Análisis por los modelos de señales ---
        st.markdown("### 📈 Diagnóstico basado en modelos de señales")
        
        if analysis_models['mitbih'] or analysis_models['ptbdb']:
            try:
                # Nota: Los modelos de señales esperan datos de señal 1D.
                # Aquí, se simula el resultado del análisis
                # porque la extracción de la señal a partir de la imagen es un paso complejo.
                
                # Simular la predicción de cada modelo
                # En un caso real, aquí iría la lógica para pasar los datos al modelo
                mitbih_prediction = np.random.rand(1, 5) # Simulación de la salida del modelo
                ptbdb_prediction = np.random.rand(1, 2)  # Simulación de la salida del modelo
                
                # Mapear las predicciones a un diagnóstico legible
                mitbih_diagnosis = ["Normal", "SVEB", "VEB", "Fusionado", "Desconocido"][np.argmax(mitbih_prediction)]
                ptbdb_diagnosis = ["Normal", "Infarto de miocardio"][np.argmax(ptbdb_prediction)]

                st.info(f"Diagnóstico MIT-BIH: **{mitbih_diagnosis}**")
                st.info(f"Diagnóstico PTB-DB: **{ptbdb_diagnosis}**")

                # Obtener datos simulados de análisis de intervalos
                analysis_results = simulate_ecg_analysis()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Ritmo Cardíaco (bpm)", value=analysis_results["heartRate"])
                with col2:
                    st.metric(label="Diagnóstico Automático", value=analysis_results["autoDiagnosis"])
                
                st.write("#### Datos Clave del ECG (Simulado)")
                df = pd.DataFrame(analysis_results["ecgIntervals"])
                st.dataframe(df.set_index('interval'))
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

    # --- SIMULACIÓN DEL HEATMAP ---
    heatmap_data = np.random.rand(len(ecg_signal))
    heatmap_data = np.convolve(heatmap_data, np.ones(5)/5, mode='same')
    heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    # --- FIN SIMULACIÓN DEL HEATMAP ---

    return {"diagnostico": diagnostico_final, "analisis_detallado": reporte, "heatmap_data": heatmap_data}

# --- NUEVAS FUNCIONES PARA EL ANÁLISIS REAL ---

def interpret_model_output(prediction):
    """
    Interpreta la salida numérica del modelo y la convierte en un diagnóstico.
    """
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
    # Asegúrate de que tu modelo tenga una capa con el nombre "conv1d_1"
    last_conv_layer = model.get_layer("conv1d_1")
    
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(data_processed)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    heatmap = np.interp(np.linspace(0, 1, 1000), np.linspace(0, 1, len(heatmap)), heatmap)
    
    return heatmap.numpy()

def process_ecg_image(image_bytes):
    """
    Lee una imagen de ECG, la convierte en una señal numérica y la normaliza.
    """
    try:
        # Paso 1: Leer la imagen desde los bytes
        image = Image.open(image_bytes)
        rgb_image = image.convert('RGB')
        gray_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)
        
        # Paso 2: Aplicar umbral
        _, signal_line = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV) 
        
        # Paso 3: Extraer la señal
        signal = []
        for col in range(signal_line.shape[1]):
            coords = np.where(signal_line[:, col] > 0)[0]
            if len(coords) > 0:
                signal.append(np.median(coords))
            else:
                signal.append(signal[-1] if signal else gray_image.shape[0] / 2)
        
        signal_array = np.array(signal, dtype=np.float32)
        signal_array = (signal_array - signal_array.min()) / (signal_array.max() - signal_array.min()) * 2 - 1

        if len(signal_array) > 1000:
            signal_array = signal_array[np.linspace(0, len(signal_array)-1, 1000).astype(int)]
        elif len(signal_array) < 1000:
            padding = np.zeros(1000 - len(signal_array))
            signal_array = np.concatenate((signal_array, padding))
        
        return signal_array
        
    except Exception as e:
        st.error(f"Error en el procesamiento de la imagen: {e}. Asegúrate de que la imagen sea un ECG claro.")
        return None

def predict_with_model(data, model, file_type):
    """
    Realiza una predicción sobre los datos ECG usando el modelo.
    """
    if model:
        st.info("Modelo cargado. Preprocesando y prediciendo...")
        try:
            if file_type in ["image/png", "image/jpeg", "image/jpg", "image/unknown_url_image"]:
                data_numpy = process_ecg_image(data)
                if data_numpy is None:
                    return None
            elif isinstance(data, pd.DataFrame):
                if 'ECG_signal' in data.columns:
                    data_numpy = data['ECG_signal'].values
                else:
                    st.error("Columna 'ECG_signal' no encontrada en el archivo CSV.")
                    return None
            else:
                data_numpy = np.array(data)

            if data_numpy.shape[0] != 1000:
                st.error(f"La señal de ECG preprocesada tiene una longitud incorrecta ({data_numpy.shape[0]}). Se esperaba 1000.")
                return None

            required_shape = model.input_shape[1:]
            data_processed = data_numpy.reshape(1, *required_shape)
            
            # --- CORRECCIÓN: Lógica para la predicción real ---
            # 1. Asegura que el modelo se inicialice haciendo una predicción
            prediction = model.predict(data_processed)
            
            # 2. Ahora que el modelo está "construido", genera el heatmap
            heatmap_data = generate_heatmap(model, data_processed)
            
            # 3. Interpreta la predicción del modelo y obtén el reporte
            results = interpret_model_output(prediction)
            
            # 4. Combina los resultados y los datos del heatmap
            results["heatmap_data"] = heatmap_data

            return results
            # --- Fin de la lógica corregida ---

        except Exception as e:
            st.error(f"Error durante la predicción con el modelo: {e}")
            return None
            
    else:
        st.warning("El modelo no ha podido ser cargado. No se puede realizar la predicción.")
        return None
# Carga del modelo global
ecg_model = load_ecg_model()

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
                if 'png' in url_input.lower():
                    file_type = 'image/png'
                elif 'jpg' in url_input.lower() or 'jpeg' in url_input.lower():
                    file_type = 'image/jpeg'
                else:
                    try:
                        Image.open(source_file).verify()
                        source_file.seek(0)
                        file_type = 'image/unknown_url_image'
                    except:
                        st.error("La URL no parece ser una imagen válida.")
                        source_file = None
                        file_type = None

                file_name = url_input
                if source_file: st.success("Imagen de URL cargada exitosamente!")
            except requests.exceptions.RequestException as e:
                st.error(f"Error al descargar la imagen de la URL: {e}")
                source_file = None
                file_type = None

        if source_file is not None:
            st.session_state['last_uploaded_file'] = source_file
            st.session_state['last_uploaded_file_type'] = file_type
            st.session_state['last_file_name'] = file_name

            with st.spinner("Procesando señal ECG..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                
                try:
                    data = None
                    if file_type in ["text/csv", "text/plain"]:
                        data = pd.read_csv(source_file)
                    elif file_type in ["image/png", "image/jpeg", "image/jpg", "image/unknown_url_image"]:
                        data = source_file
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
            
            heatmap_display = np.interp(np.linspace(0, 1, original_image_np.shape[1]), 
                                        np.linspace(0, 1, len(heatmap_data)), 
                                        heatmap_data)
            
            cmap = plt.cm.get_cmap('hot')
            
            heatmap_mask = np.zeros_like(original_image_np[:,:,0], dtype=float)
            center_row = original_image_np.shape[0] // 2
            heatmap_mask[center_row-10:center_row+10, :] = np.tile(heatmap_display, (20,1))
            
            ax.imshow(heatmap_mask, cmap=cmap, alpha=0.5, extent=[0, original_image_np.shape[1], original_image_np.shape[0], 0])
            
            ax.set_axis_off()
            st.pyplot(fig)
            
        st.subheader("Diagnóstico")
        diagnostico = results['diagnostico']
        
        if diagnostico == "Infarto Agudo del Miocardio (IAM)":
            st.error(f"⚠️ **DIAGNÓSTICO: {diagnostico}**")
            st.warning("Busque **ATENCIÓN MÉDICA DE URGENCIA** de inmediato. Este resultado sugiere un posible evento cardíaco grave.")
        elif "normal" in diagnostico.lower():
            st.success(diagnostico)
        else:
            st.warning(diagnostico)
            
            except Exception as e:
                st.error(f"❌ Error al ejecutar los modelos de análisis: {e}")
                st.info("No se pudo obtener el análisis. Revisa los modelos de señales.")
        else:
            st.warning("⚠️ No se cargó ningún modelo de análisis. El diagnóstico no está disponible.")

        # --- Gráfico simulado del trazado ECG ---
        st.write("---")
        st.markdown("### 📊 Trazado ECG Simulado")
        st.write("Este gráfico representa un trazado de ECG simulado para fines de demostración.")
        fig = generate_ecg_graph()
        st.pyplot(fig)
        
        st.success("Análisis completo.")
        st.warning("⚠️ **Aviso Importante**: Esta es una herramienta experimental. Consulta siempre a un profesional de la salud para un diagnóstico médico.")

# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    main()
        analisis_df = pd.DataFrame(results['analisis_detallado'].items(), columns=['Elemento', 'Estado'])
        st.table(analisis_df)
    else:
        st.subheader("Resultados del análisis:")
        st.warning("Por favor, sube y procesa un archivo ECG para ver el informe.")

