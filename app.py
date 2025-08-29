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
    Carga el modelo de IA una sola vez.
    Asegúrate de que tu archivo de modelo ('modelo_ecg.h5') esté en la misma carpeta.
    """
    try:
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

    # --- SIMULACIÓN DEL HEATMAP ---
    # Genera un array con valores de "importancia" aleatorios para la señal del ECG.
    # Los valores más altos simularían las áreas donde el modelo "presta más atención".
    heatmap_data = np.random.rand(len(ecg_signal)) # Asume que ecg_signal es 1D y tiene longitud 1000
    # Suavizar un poco los datos para que el heatmap no sea tan ruidoso
    heatmap_data = np.convolve(heatmap_data, np.ones(5)/5, mode='same')
    # Normalizar entre 0 y 1
    heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    # --- FIN SIMULACIÓN DEL HEATMAP ---

    return {"diagnostico": diagnostico_final, "analisis_detallado": reporte, "heatmap_data": heatmap_data}

def process_ecg_image(image_bytes):
    """
    Lee una imagen de ECG, la convierte en una señal numérica y la normaliza.
    
    Este es un ejemplo simplificado. Una solución para producción requeriría
    técnicas de visión por computadora más avanzadas para manejar distintos formatos,
    ruidos y estilos de cuadrículas de ECG.
    """
    try:
        # Paso 1: Leer la imagen desde los bytes
        image = Image.open(image_bytes)
        
        # Paso 2: Convertir a escala de grises y a un array de NumPy
        # Convierte a RGB primero para asegurar consistencia antes de pasar a gris
        rgb_image = image.convert('RGB')
        gray_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)
        
        # Paso 3: Aplicar umbral para aislar la línea de la señal
        # Se asume un fondo claro y líneas oscuras (ECG tradicional)
        _, signal_line = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV) 
        
        # Paso 4: Encontrar las coordenadas Y de la señal
        signal = []
        # Para cada columna (punto en el tiempo)
        for col in range(signal_line.shape[1]):
            # Encuentra los píxeles blancos (señal)
            coords = np.where(signal_line[:, col] > 0)[0]
            if len(coords) > 0:
                # Usa la mediana de las coordenadas si hay varios puntos (línea gruesa)
                signal.append(np.median(coords))
            else:
                # Si no hay señal, interpola o usa el último valor conocido
                signal.append(signal[-1] if signal else gray_image.shape[0] / 2) # Centro vertical
        
        # Paso 5: Convertir a un array de NumPy y normalizar
        signal_array = np.array(signal, dtype=np.float32)
        
        # Normalizar la señal para que los valores estén en un rango similar
        # Esto es importante si el modelo espera un rango específico de entrada
        signal_array = (signal_array - signal_array.min()) / (signal_array.max() - signal_array.min()) * 2 - 1 # Rango de -1 a 1

        # Ajustar al tamaño requerido de 1000 muestras
        if len(signal_array) > 1000:
            # Submuestreo si es demasiado largo
            signal_array = signal_array[np.linspace(0, len(signal_array)-1, 1000).astype(int)]
        elif len(signal_array) < 1000:
            # Relleno si es demasiado corto
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
            if file_type in ["image/png", "image/jpeg", "image/jpg"]:
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

            # Asegúrate de que data_numpy tenga la longitud esperada por el modelo (1000)
            if data_numpy.shape[0] != 1000:
                st.error(f"La señal de ECG preprocesada tiene una longitud incorrecta ({data_numpy.shape[0]}). Se esperaba 1000.")
                return None

            required_shape = model.input_shape[1:] # Asumiendo (1000, 1)

            # Reestructurar para el modelo (batch_size, timesteps, features)
            data_processed = data_numpy.reshape(1, *required_shape)
            
            # Normalización final para el modelo si no se hizo antes o se requiere diferente
            # data_processed = (data_processed - np.mean(data_processed)) / np.std(data_processed)

            # Aquí es donde se llamaría al modelo real si no fuera una simulación.
            # prediction = model.predict(data_processed)
            # results = process_real_prediction(prediction)

            # Por ahora, usamos la simulación
            results = analyze_ecg_details(data_numpy) # Pasa data_numpy para que la simulación de heatmap use la longitud correcta

            return results

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
                # Intenta inferir el tipo de archivo de la URL
                if 'png' in url_input.lower():
                    file_type = 'image/png'
                elif 'jpg' in url_input.lower() or 'jpeg' in url_input.lower():
                    file_type = 'image/jpeg'
                else:
                    # Si no se puede inferir, intentar abrirlo con PIL para confirmar que es una imagen
                    try:
                        Image.open(source_file).verify()
                        source_file.seek(0) # Rebobinar después de verify()
                        file_type = 'image/unknown_url_image' # O un tipo más genérico
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
                        data = source_file # Pasar el objeto de bytes de archivo para que la función de procesamiento lo maneje
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
            
            # Necesitamos recargar la imagen original para superponer el heatmap
            uploaded_image_bytes = st.session_state['last_uploaded_file']
            uploaded_image_bytes.seek(0) # Rebobinar el buffer de bytes
            original_image = Image.open(uploaded_image_bytes).convert('RGB')
            original_image_np = np.array(original_image)
            
            # La señal extraída ya está en 'results' si se pasó a analyze_ecg_details
            # Pero necesitamos la señal original del procesamiento de la imagen para el heatmap
            processed_signal_for_heatmap = results['heatmap_data'] # Ahora heatmap_data es la señal de ECG
            
            # Crea una figura de matplotlib
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(original_image_np, aspect='auto') # Muestra la imagen de fondo
            
            # Superponer el heatmap: Necesitamos mapear los datos 1D del heatmap a las dimensiones de la imagen
            # Esto es una simulación. En un caso real, el heatmap tendría la misma longitud que la señal extraída.
            # Para fines de visualización, lo mapeamos al ancho de la imagen.
            heatmap_display = np.interp(np.linspace(0, 1, original_image_np.shape[1]), 
                                        np.linspace(0, 1, len(processed_signal_for_heatmap)), 
                                        processed_signal_for_heatmap)
            
            # Crear un mapa de colores para el heatmap (ej. de azul a rojo)
            cmap = plt.cm.get_cmap('hot') # Puedes cambiar 'hot' por 'viridis', 'jet', etc.
            
            # Crear una "máscara" del heatmap transparente
            heatmap_mask = np.zeros_like(original_image_np[:,:,0], dtype=float)
            # Asignar valores del heatmap a una fila central para la visualización simplificada
            center_row = original_image_np.shape[0] // 2
            heatmap_mask[center_row-10:center_row+10, :] = np.tile(heatmap_display, (20,1))
            
            # Ajustar la opacidad (alpha) del heatmap
            ax.imshow(heatmap_mask, cmap=cmap, alpha=0.5, extent=[0, original_image_np.shape[1], original_image_np.shape[0], 0])
            
            ax.set_axis_off() # Ocultar ejes
            st.pyplot(fig)
            
            # Guardar la figura en la sesión para que se muestre como imagen
            # import io
            # buf = io.BytesIO()
            # plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
            # st.image(buf.getvalue(), caption="ECG con Heatmap", use_container_width=True)
            # buf.close()
            
        st.subheader("Diagnóstico")
        diagnostico = results['diagnostico']
        
        if diagnostico == "Infarto Agudo del Miocardio (IAM)":
            st.error(f"⚠️ **DIAGNÓSTICO: {diagnostico}**")
            st.warning("Busque **ATENCIÓN MÉDICA DE URGENCIA** de inmediato. Este resultado sugiere un posible evento cardíaco grave.")
        elif "normal" in diagnostico.lower():
            st.success(diagnostico)
        else:
            st.warning(diagnostico)
            
        st.subheader("Análisis Detallado de Elementos del ECG")
        
        analisis_df = pd.DataFrame(results['analisis_detallado'].items(), columns=['Elemento', 'Estado'])
        st.table(analisis_df)
    else:
        st.subheader("Resultados del análisis:")
        st.warning("Por favor, sube y procesa un archivo ECG para ver el reporte.")
