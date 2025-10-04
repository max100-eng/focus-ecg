# plumber.R

# --- Proceso de Arranque ---
# Carga las librerías necesarias para tu API
# 'plumber' es para la API, y 'randomForest'/'ranger' son para tu modelo.
library(plumber)
library(readr)
library(randomForest) # O la librería que uses para tu modelo (ej. ranger)

# Carga el modelo una sola vez, al iniciar la API.
# Esto hace que el modelo esté disponible para todas las solicitudes.
modelo_ecg <- readRDS("modelo_ecg.rds")

# --- Endpoint de la API ---
# El decorador '@post' indica que este endpoint solo acepta solicitudes POST.
# '/prediccion_ecg' es el nombre del endpoint.
# '@param' especifica el nombre del parámetro de entrada, que será una lista.
# '@serializer json' asegura que la respuesta se devuelva en formato JSON.

#* @post /prediccion_ecg
#* @param datos:list
#* @serializer json
function(datos) {
  # Convierte la lista de datos recibida a un data.frame
  df_input <- as.data.frame(datos)
  
  # Realiza la predicción usando el modelo ya cargado
  prediccion <- predict(modelo_ecg, newdata = df_input)
  
  # Devuelve la predicción como una lista para ser serializada en JSON
  list(prediccion = prediccion)
}