import streamlit as st
import joblib
import numpy as np

# --- Función para cargar modelos de forma eficiente ---
@st.cache_resource
def load_models():
    """Carga los modelos entrenados (.joblib)."""
    modelos = {}
    try:
        modelos["Dólar"] = joblib.load('modelo_dolar.joblib')
        # Simular la carga de los otros modelos
        # modelos["Glucosa"] = joblib.load('modelo_glucosa.joblib') 
        # modelos["Energía"] = joblib.load('modelo_energia.joblib') 
        
        # En caso de no existir los otros modelos (para propósitos de prueba)
        # se usará el del Dólar como placeholder.
        if "Glucosa" not in modelos:
            modelos["Glucosa"] = modelos["Dólar"]
        if "Energía" not in modelos:
            modelos["Energía"] = modelos["Dólar"]
            
        return modelos
    except FileNotFoundError as e:
        st.error(f"Error al cargar el modelo: {e}. Asegúrate de que los archivos .joblib existan.")
        return None

modelos = load_models()

# --- Configuración de la Aplicación ---
st.title("💸 Sistema de Predicción ML")
st.markdown("Utilice los modelos de Regresión Lineal entrenados para obtener predicciones.")

# Selector de Ejercicio
ejercicio_seleccionado = st.selectbox(
    "Selecciona el Escenario de Predicción:",
    ("Dólar", "Glucosa", "Energía")
)

st.header(f"Predicción: {ejercicio_seleccionado}")
st.markdown("---")

if modelos:
    modelo_actual = modelos[ejercicio_seleccionado]
    variables = {}
    
    # --- Definición de Inputs por Escenario ---
    
    if ejercicio_seleccionado == "Dólar":
        st.subheader("Ingresa los valores (Dólar):")
        variables['Dia'] = st.number_input("1. Día:", min_value=1, value=500, step=1)
        variables['Inflacion'] = st.number_input("2. Inflación diaria:", min_value=0.0, value=0.02, format="%.4f")
        variables['Tasa_interes'] = st.number_input("3. Tasa de Interés diaria:", min_value=0.0, value=5.0, format="%.2f")
        feature_order = ['Dia', 'Inflacion', 'Tasa_interes']
        
    elif ejercicio_seleccionado == "Glucosa":
        # Estas variables son hipotéticas para el ejercicio de Glucosa
        st.subheader("Ingresa los valores (Glucosa):")
        variables['Edad'] = st.number_input("1. Edad (años):", min_value=1, value=45, step=1)
        variables['IMC'] = st.number_input("2. Índice de Masa Corporal (IMC):", min_value=10.0, value=25.0, format="%.1f")
        variables['Hormona_x'] = st.number_input("3. Nivel de Hormona X:", min_value=0.0, value=10.0, format="%.1f")
        feature_order = ['Edad', 'IMC', 'Hormona_x']
        
    elif ejercicio_seleccionado == "Energía":
        # Estas variables son hipotéticas para el ejercicio de Energía
        st.subheader("Ingresa los valores (Energía):")
        variables['Temperatura'] = st.number_input("1. Temperatura Ambiente (°C):", min_value=0.0, value=25.0, format="%.1f")
        variables['Hum_Relativa'] = st.number_input("2. Humedad Relativa (%):", min_value=0.0, value=60.0, format="%.1f")
        variables['Consumo_Previa'] = st.number_input("3. Consumo Hora Previa (kWh):", min_value=0.0, value=5.0, format="%.1f")
        feature_order = ['Temperatura', 'Hum_Relativa', 'Consumo_Previa']
        
    # --- Botón de Predicción y Lógica ---
    if st.button("Obtener Predicción"):
        try:
            # Reorganizar los datos ingresados en el orden correcto para el modelo
            input_data = np.array([[variables[f] for f in feature_order]])
            
            # Realizar la predicción
            prediccion = modelo_actual.predict(input_data)[0]
            
            unidad = " unidades monetarias"
            if ejercicio_seleccionado == "Glucosa":
                unidad = " mg/dL"
            elif ejercicio_seleccionado == "Energía":
                unidad = " kWh"
            
            st.success(f"La predicción para **{ejercicio_seleccionado}** es:")
            st.balloons()
            st.markdown(f"## **{prediccion:,.2f}{unidad}**")
            
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")

# --- Instrucciones para Ejecutar la Interfaz ---
st.sidebar.markdown("### ⚙️ Instrucciones")
st.sidebar.markdown("1. Guarda el código arriba como `app.py`.")
st.sidebar.markdown("2. Asegúrate de tener los archivos `modelo_X.joblib` en la misma carpeta.")
st.sidebar.markdown("3. Ejecuta en tu terminal:")
st.sidebar.code("streamlit run app.py")