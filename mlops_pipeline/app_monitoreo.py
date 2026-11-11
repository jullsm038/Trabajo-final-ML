"""
app_monitoreo.py
Tablero en Streamlit para ver el monitoreo del modelo (drift).

ÍNDICE
  1. Configuración y carga de datos
  2. Resumen general
  3. Drift en el tiempo (mapa de calor)
  4. Detalle de un mes
  5. Alertas y recomendación

Cómo encenderla (desde la carpeta mlops_pipeline):
    streamlit run app_monitoreo.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from model_monitoring import (construir_tabla_monitoreo, dividir_historico_y_periodos,
                              drift_en_el_tiempo, generar_alertas)

# =====================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# =====================================================================
st.set_page_config(page_title="Monitoreo del modelo", layout="wide")
st.title("Monitoreo del modelo de crédito")
st.caption("Compara los primeros 3 meses (referencia) contra cada mes siguiente.")


@st.cache_data
def cargar():
    tabla = construir_tabla_monitoreo()
    historico, periodos, meses_historico = dividir_historico_y_periodos(tabla)
    drift = drift_en_el_tiempo(tabla)
    return tabla, historico, meses_historico, drift


tabla, historico, meses_historico, drift = cargar()

# =====================================================================
# 2. RESUMEN GENERAL
# =====================================================================
col1, col2, col3 = st.columns(3)
col1.metric("Registros totales", f"{len(tabla):,}")
col2.metric("Registros de referencia", f"{len(historico):,}")
col3.metric("Meses analizados", drift["periodo"].nunique())

resumen = (drift.groupby("periodo").agg(
    criticas=("nivel", lambda s: int((s == "Crítico").sum())),
    alertas=("nivel", lambda s: int((s == "Alerta").sum()))))
st.subheader("Variables en estado crítico y en alerta por mes")
st.bar_chart(resumen)

# =====================================================================
# 3. DRIFT EN EL TIEMPO
# =====================================================================
st.header("Drift en el tiempo")
st.write("Cada celda muestra el PSI de una variable en un mes. Más intenso = más cambio. "
         "Menos de 0,10 estable; 0,10 a 0,25 alerta; más de 0,25 crítico.")
mapa = drift.pivot(index="variable", columns="periodo", values="psi")
figura, ejes = plt.subplots(figsize=(12, 8))
sns.heatmap(mapa, cmap="YlOrRd", vmax=0.5, annot=False, ax=ejes)
st.pyplot(figura)

# =====================================================================
# 4. DETALLE DE UN MES
# =====================================================================
st.header("Detalle de un mes")
mes = st.selectbox("Elige un mes", sorted(drift["periodo"].unique()), index=len(resumen) - 1)
detalle = drift[drift["periodo"] == mes]
st.dataframe(detalle[["variable", "psi", "ks", "js", "chi2_pvalor", "nivel"]].round(4),
             use_container_width=True)

# =====================================================================
# 5. ALERTAS Y RECOMENDACIÓN
# =====================================================================
st.header("Alertas")
for tipo, mensaje in generar_alertas(detalle):
    if tipo == "Crítico":
        st.error(f"{tipo}: {mensaje}")
    elif tipo == "Alerta":
        st.warning(f"{tipo}: {mensaje}")
    else:
        st.info(f"{tipo}: {mensaje}")