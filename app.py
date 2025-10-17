# Importamos las librerías necesarias:

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from groq import Groq 

# --- 1. Configuración Inicial y Carga de Modelos ---

# Cargar el modelo de clasificación de Pokémon
# Asegúrate de que los archivos del modelo estén en la misma carpeta que este script
# o proporciona la ruta completa.

@st.cache_resource      # Cachear el modelo para que no se cargue cada vez
def load_prediction_model():
    model = tf.keras.models.load_model(r'pokedex_model.h5')
    return model

# Cargar los nombres de las clases (los Pokémon)
@st.cache_data
def load_class_names():
    class_names = np.load(r'pokemon_class_names.npy', allow_pickle=True)
    return class_names

model = load_prediction_model()
class_names = load_class_names()

# Configuración del cliente de Groq con la API Key
try:
    # Para despliegue en Streamlit Cloud, usa los Secrets en Ajustes al hacer el deploy de la app
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    # Para pruebas locales, podes poner la clave acá (no recomendado para producción)
    # O para manejar el error si la clave no está disponible.
    # Mensaje de error (comentar si se despliega de manera local):
    st.error("API Key de Groq no encontrada. Por favor, configúrala en los secretos de Streamlit.")
    client = None


# --- 2. Funciones de la Aplicación ---

def predict_pokemon(image):
    """
    Toma una imagen, la preprocesa y usa el modelo para predecir el Pokémon.
    """
    # Preprocesamiento de la imagen para que coincida con la entrada del modelo:
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)   # Crea un batch
    img_array /= 255.0      # Normaliza

    # Realiza la predicción:
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    predicted_pokemon = class_names[predicted_class_index]

    return predicted_pokemon, confidence

def get_pokemon_info(pokemon_name, user_question):
    """
    Llama a la API de Groq para obtener información sobre el Pokémon.
    """
    if not client:
        return "El cliente de Groq no está configurado. Revisa la API Key."

    prompt = f"""
    Eres una Pokédex experta. Responde la siguiente pregunta sobre {pokemon_name} de manera concisa y amigable.
    Pregunta: "{user_question}"
    """
    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Usamos un modelo de Groq
            messages=[
                {"role": "system", "content": "Eres una Pokédex experta."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al obtener información de la Pokédex: {e}"
    

# --- 3. Diseño de la Interfaz de Streamlit ---

st.title("Pokédex Inteligente 🤖")
st.write("¡Toma la imagen de un Pokémon y haz una pregunta sobre él!")


# Carga de la imagen por parte del usuario
uploaded_file = st.file_uploader("Sube una imagen de un Pokémon...", type=["jpg"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada!', use_container_width=False)
    st.write("")
    st.write("Identifica al Pokémon presionando el botón debajo...")

    # Realizar la predicción al presionar un botón
    if st.button('¿Quién es ese Pokémon?'):
        pokemon_name, confidence = predict_pokemon(image)
        st.session_state.pokemon_name = pokemon_name # Guardar el nombre en el estado de la sesión
        st.success(f"¡Pokémon identificado! Es un **{pokemon_name}** con una confianza del {confidence:.2f}%.")

# Sección para preguntas y respuestas
if 'pokemon_name' in st.session_state:
    st.header(f"Pregúntale a la Pokédex sobre {st.session_state.pokemon_name}")
    user_question = st.text_input("Ej: ¿Cuál es su evolución?", key="question_input")

    if st.button("Obtener Respuesta"):
        if user_question:
            with st.spinner("Consultando base de datos de la Pokédex..."):
                answer = get_pokemon_info(st.session_state.pokemon_name, user_question)
                st.write(answer)
        else:
            st.warning("Por favor, escribe una pregunta:")
