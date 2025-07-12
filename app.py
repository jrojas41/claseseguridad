#App para predicción de problema cardiaco 
#NO CORRER CON EL BOTON DE RUN **********************************************

'''
se ejecuta primero cargan las bibliotecas requeridas con
pip install -r requirements.txt (tenga creado ese archivo)

se ejecuta la aplicacion con
streamlit run app.py desde la linea del terminal o de su powershell

'''

import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Cargar el modelo y el escalador
try:
    svc_model = joblib.load('svc_model.jb')
    scaler = joblib.load('scaler.jb')
except FileNotFoundError:
    st.error("Error: Asegúrate de tener los archivos 'svc_model.jb' y 'scaler.jb' en el mismo directorio.")
    st.stop()

# Cargar las imágenes
try:
    cabezote_img = Image.open('cabezote.jpg')
    nosufre_img = Image.open('Nosufre.jpg')
    sisufre_img = Image.open('Sisufre.jpg')
except FileNotFoundError:
    st.warning("Advertencia: No se encontraron algunas imágenes (cabezote.jpg, Nosufre.jpg, Sisufre.jpg). Continúa sin ellas.")
    cabezote_img = None
    nosufre_img = None
    sisufre_img = None


# Título y banner
if cabezote_img:
    st.image(cabezote_img, use_column_width=True)
st.title("Modelo IA para predicción de problemas cardiacos")

# Resumen del modelo
st.write("""
Este modelo de Inteligencia Artificial utiliza un clasificador de Máquinas de Vectores de Soporte (SVC)
para predecir la probabilidad de que un paciente sufra de problemas cardiacos
basado en su edad y nivel de colesterol. Los datos de entrada son escalados
para mejorar el rendimiento del modelo.

**Cómo funciona:**
El modelo ha sido entrenado con datos históricos de pacientes para identificar patrones
entre la edad, el colesterol y la presencia de problemas cardiacos. Al ingresar tus datos,
el modelo evalúa estos patrones y predice si tienes un mayor o menor riesgo.
""")

# Sidebar para la entrada de usuario
st.sidebar.header("Introduce los datos del paciente")

edad = st.sidebar.slider("Edad:", min_value=20, max_value=80, value=20, step=1)
colesterol = st.sidebar.slider("Colesterol:", min_value=120, max_value=600, value=200, step=10)

# Preparar los datos para la predicción
data = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})

# Escalar los datos de entrada
scaled_data = scaler.transform(data)
scaled_data_df = pd.DataFrame(scaled_data, columns=['edad', 'colesterol'])

# Realizar la predicción
prediction = svc_model.predict(scaled_data_df)

# Mostrar el resultado
st.header("Resultado de la predicción")

if prediction[0] == 0:
    st.markdown(
        """
        <div style="background-color:#90EE90; padding: 10px; border-radius: 5px;">
            <h3 style="color:black; text-align:center;">¡No sufrirá del corazón! 😊</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    if nosufre_img:
        st.image(nosufre_img, use_column_width=True)
else:
    st.markdown(
        """
        <div style="background-color:#F08080; padding: 10px; border-radius: 5px;">
            <h3 style="color:black; text-align:center;">Sufrirá del corazón 😥</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    if sisufre_img:
        st.image(sisufre_img, use_column_width=True)

# Información del autor
st.markdown("---")
st.write("Elaborado por: juan rojas © Unab 2025")
