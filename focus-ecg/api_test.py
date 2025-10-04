import requests
import json

# URL completa del endpoint.
# Esto incluye la ruta de la API y el endpoint para la predicción.
api_url = "https://2ldfc4-massimo-barbetta.shinyapps.io/focus-ecg-api/prediccion_ecg"

# Los datos que se enviarán a la API.
# 'frecuencia_cardiaca' y 'variabilidad_rr' deben coincidir
# exactamente con los nombres de las variables que espera tu modelo de R.
datos_a_enviar = {
    "datos": [
        {
            "frecuencia_cardiaca": 75,
            "variabilidad_rr": 120
        }
    ]
}

# Realiza la solicitud POST y envía los datos en formato JSON.
try:
    response = requests.post(api_url, json=datos_a_enviar)

    # Verifica si la solicitud fue exitosa (código 200).
    if response.status_code == 200:
        # La API devuelve un JSON que se convierte a un diccionario de Python.
        resultado = response.json()
        print("Predicción de la API:", resultado)
    else:
        # Si la respuesta no es 200, imprime el código de error.
        print(f"Error en la API. Código de estado: {response.status_code}")
        print("Respuesta del servidor:", response.text)
        
except requests.exceptions.RequestException as e:
    # Captura cualquier error de conexión.
    print(f"Error de conexión: {e}")