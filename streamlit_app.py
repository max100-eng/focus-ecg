import streamlit as st
import requests

# 1. Configuración de la API (DEBES REEMPLAZAR ESTA URL)
# Esta es la URL pública que obtuviste después de desplegar tu API de Plumber en R.
API_URL = "https://tu-url-de-api-desplegada.com/prediccion_ecg" 
# NOTA: Asegúrate de que el path del endpoint (/prediccion_ecg) sea el correcto.

st.set_page_config(page_title="Focus-ECG Predictor", layout="centered")

st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #007bff;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Focus-ECG: Herramienta de Predicción</p>', unsafe_allow_html=True)
st.write("Introduce los parámetros del ECG para obtener la predicción del modelo de R.")

# --- Recolección de la entrada del usuario ---
st.subheader("Parámetros de Entrada")
col1, col2 = st.columns(2)

with col1:
    frec_cardiaca = st.number_input(
        "Frecuencia Cardíaca (lpm):", 
        min_value=30.0, 
        max_value=200.0, 
        value=85.0, 
        step=0.1,
        help="Introduce la frecuencia cardíaca promedio en latidos por minuto."
    )

with col2:
    var_rr = st.number_input(
        "Variabilidad RR (ms):", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.05, 
        step=0.001,
        help="Introduce el valor de la variabilidad del intervalo R-R."
    )

st.markdown("---")

# --- Lógica de Llamada a la API ---

if st.button("Obtener Predicción", type="primary"):
    # 2. Construir el diccionario de Python
    # Las CLAVES deben coincidir EXACTAMENTE con los nombres de las columnas que tu modelo de R espera.
    datos_para_api = {
        "frecuencia_cardiaca": frec_cardiaca,
        "variabilidad_rr": var_rr
    }

    try:
        with st.spinner('Contactando con la API de R...'):
            # 3. Enviar la solicitud POST
            # El argumento 'json=...' serializa el diccionario a JSON automáticamente
            respuesta = requests.post(API_URL, json=datos_para_api)
        
        # 4. Procesar la respuesta
        if respuesta.status_code == 200:
            prediccion_json = respuesta.json()
            
            # Asumiendo que la respuesta es {"prediccion": [valor]} o similar
            # Accede a la clave de predicción (puede ser 'prediccion', 'resultado', etc., ¡revisa tu API de R!)
            resultado = prediccion_json.get('prediccion', 'Resultado no encontrado')

            st.success("✅ Predicción Obtenida Exitosamente")
            st.metric(
                label="Resultado del ECG", 
                value=f"{resultado}",
                delta="Modelo de R"
            )
            st.json(prediccion_json) # Muestra la respuesta RAW para debug

        else:
            st.error(f"❌ Error en la API: Código {respuesta.status_code}")
            st.code(respuesta.text, language='text')
            
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Error de Conexión: Asegúrate de que la API de Plumber está funcionando y la URL es correcta.")
        st.code(str(e), language='text')

st.markdown("---")
st.caption("Esta aplicación interactúa con un modelo de Machine Learning alojado como una API RESTful en R/Plumber.")
