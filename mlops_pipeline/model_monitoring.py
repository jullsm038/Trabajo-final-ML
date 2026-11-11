"""
model_monitoring.py
Monitoreo del modelo y detección de data drift.

El "data drift" es cuando los datos nuevos dejan de parecerse a los datos con los
que se entrenó el modelo. Si pasa, el modelo puede empezar a fallar sin que nadie lo note.

ÍNDICE
  1. Datos fijos: rutas, periodos y umbrales
  2. Tabla de monitoreo (datos + pronósticos del modelo)
  3. Métricas de drift: KS, PSI, Jensen-Shannon y Chi-cuadrado
  4. Comparar dos periodos (histórico vs actual)
  5. Muestreo periódico: drift mes a mes
  6. Alertas y recomendaciones
  7. Gráficas
  8. Función principal

Se ejecuta con:  python mlops_pipeline/model_monitoring.py
"""
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # guarda las gráficas como imagen, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon

from ft_engineering import (COLUMNAS_CATEGORICAS, COLUMNAS_DINERO, COLUMNAS_NUMERICAS_SIMPLES,
                            COLUMNAS_RAZONES, cargar_datos, crear_variables, limpiar_datos)

warnings.filterwarnings("ignore")


# =====================================================================
# 1. DATOS FIJOS
# =====================================================================
CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "mejor_modelo.joblib"
CARPETA_RESULTADOS = CARPETA / "resultados"

# Periodo histórico: con qué se compara. Los meses siguientes se revisan uno por uno.
MESES_REFERENCIA = 3
MIN_REGISTROS_PERIODO = 100   # un mes con menos registros no se analiza (poco confiable)
N_BINS = 10                   # grupos para PSI y Jensen-Shannon

# Umbrales del semáforo
PSI_ALERTA, PSI_CRITICO = 0.10, 0.25
KS_ALERTA, KS_CRITICO = 0.10, 0.20

# Variables que se vigilan (mes_prestamo no se vigila: cambia siempre por definición)
VARIABLES_BINARIAS = ["tiene_mora"]
VARIABLES_NUMERICAS = [c for c in COLUMNAS_DINERO + COLUMNAS_RAZONES + COLUMNAS_NUMERICAS_SIMPLES
                       if c not in ["mes_prestamo"] + VARIABLES_BINARIAS]
VARIABLES_CATEGORICAS = COLUMNAS_CATEGORICAS + VARIABLES_BINARIAS
COLUMNAS_MODELO = (COLUMNAS_DINERO + COLUMNAS_RAZONES + COLUMNAS_NUMERICAS_SIMPLES + COLUMNAS_CATEGORICAS)


# =====================================================================
# 2. TABLA DE MONITOREO
# Une los datos con los pronósticos del modelo y marca el periodo (mes).
# =====================================================================
def construir_tabla_monitoreo(ruta_modelo=RUTA_MODELO):
    datos = crear_variables(limpiar_datos(cargar_datos()))
    modelo = joblib.load(ruta_modelo)

    tabla = datos[COLUMNAS_MODELO].copy()
    columna = list(modelo.classes_).index(0)
    tabla["prob_no_pago"] = modelo.predict_proba(tabla)[:, columna]   # probabilidad de NO pagar
    tabla["prediccion"] = modelo.predict(tabla)
    tabla["periodo"] = pd.to_datetime(datos["fecha_prestamo"]).dt.to_period("M").astype(str)
    return tabla


# =====================================================================
# 3. MÉTRICAS DE DRIFT
# Todas comparan una serie "histórica" con una "actual".
# =====================================================================
def _proporciones_numericas(historico, actual, n_bins=N_BINS):
    """Reparte los valores en grupos (por percentiles del histórico) y cuenta qué porcentaje cae en cada uno."""
    cortes = np.unique(np.nanquantile(historico, np.linspace(0, 1, n_bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p_hist = np.histogram(historico, bins=cortes)[0] / len(historico)
    p_act = np.histogram(actual, bins=cortes)[0] / len(actual)
    return p_hist, p_act


def _proporciones_categoricas(historico, actual):
    categorias = sorted(set(historico.astype(str)) | set(actual.astype(str)))
    p_hist = historico.astype(str).value_counts(normalize=True).reindex(categorias, fill_value=0).values
    p_act = actual.astype(str).value_counts(normalize=True).reindex(categorias, fill_value=0).values
    return p_hist, p_act


def calcular_psi(p_hist, p_act):
    """Population Stability Index: <0,10 estable | 0,10 a 0,25 cambio moderado | >0,25 cambio fuerte."""
    p_hist, p_act = np.clip(p_hist, 1e-4, None), np.clip(p_act, 1e-4, None)
    return float(np.sum((p_act - p_hist) * np.log(p_act / p_hist)))


def calcular_js(p_hist, p_act):
    """Distancia de Jensen-Shannon: va de 0 (iguales) a 1 (totalmente distintas)."""
    return float(jensenshannon(p_hist, p_act, base=2))


def calcular_ks(historico, actual):
    """Kolmogorov-Smirnov: la mayor diferencia entre las dos distribuciones (0 a 1) y su p-valor."""
    resultado = stats.ks_2samp(historico, actual)
    return float(resultado.statistic), float(resultado.pvalue)


def calcular_chi2(historico, actual):
    """Chi-cuadrado para categorías: p-valor menor a 0,05 indica que las proporciones cambiaron."""
    categorias = sorted(set(historico.astype(str)) | set(actual.astype(str)))
    conteos = np.array([
        historico.astype(str).value_counts().reindex(categorias, fill_value=0).values,
        actual.astype(str).value_counts().reindex(categorias, fill_value=0).values,
    ])
    conteos = conteos[:, conteos.sum(axis=0) > 0]
    if conteos.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(conteos)[1])


def clasificar_nivel(psi, ks=None):
    """Semáforo: usa el PSI (y el KS cuando la variable es numérica) y se queda con el peor."""
    nivel = 2 if psi >= PSI_CRITICO else 1 if psi >= PSI_ALERTA else 0
    if ks is not None:
        nivel = max(nivel, 2 if ks >= KS_CRITICO else 1 if ks >= KS_ALERTA else 0)
    return ["Estable", "Alerta", "Crítico"][nivel]


# =====================================================================
# 4. COMPARAR DOS PERIODOS
# Calcula todas las métricas de cada variable: histórico vs actual.
# =====================================================================
def comparar_periodos(historico, actual):
    filas = []
    for columna in VARIABLES_NUMERICAS + ["prob_no_pago"]:
        h, a = historico[columna].dropna(), actual[columna].dropna()
        p_h, p_a = _proporciones_numericas(h.values, a.values)
        ks, ks_p = calcular_ks(h, a)
        psi = calcular_psi(p_h, p_a)
        filas.append({"variable": columna, "tipo": "numérica", "psi": psi, "ks": ks, "ks_pvalor": ks_p,
                      "js": calcular_js(p_h, p_a), "chi2_pvalor": np.nan,
                      "nivel": clasificar_nivel(psi, ks)})

    for columna in VARIABLES_CATEGORICAS + ["prediccion"]:
        h, a = historico[columna].fillna("Sin dato"), actual[columna].fillna("Sin dato")
        p_h, p_a = _proporciones_categoricas(h, a)
        psi = calcular_psi(p_h, p_a)
        filas.append({"variable": columna, "tipo": "categórica", "psi": psi, "ks": np.nan, "ks_pvalor": np.nan,
                      "js": calcular_js(p_h, p_a), "chi2_pvalor": calcular_chi2(h, a),
                      "nivel": clasificar_nivel(psi)})
    return pd.DataFrame(filas)


# =====================================================================
# 5. MUESTREO PERIÓDICO: DRIFT MES A MES
# El histórico son los primeros meses; cada mes siguiente se compara con él.
# =====================================================================
def dividir_historico_y_periodos(tabla):
    periodos = sorted(tabla["periodo"].unique())
    meses_historico = periodos[:MESES_REFERENCIA]
    historico = tabla[tabla["periodo"].isin(meses_historico)]
    return historico, [p for p in periodos if p not in meses_historico], meses_historico


def drift_en_el_tiempo(tabla):
    historico, periodos, _ = dividir_historico_y_periodos(tabla)
    resultados = []
    for periodo in periodos:
        actual = tabla[tabla["periodo"] == periodo]
        if len(actual) < MIN_REGISTROS_PERIODO:
            continue   # muestra muy pequeña: no es confiable
        r = comparar_periodos(historico, actual)
        r.insert(0, "periodo", periodo)
        r.insert(1, "registros", len(actual))
        resultados.append(r)
    return pd.concat(resultados, ignore_index=True)


# =====================================================================
# 6. ALERTAS Y RECOMENDACIONES
# =====================================================================
def generar_alertas(drift_periodo):
    """Recibe el drift de UN periodo y devuelve mensajes de alerta y sugerencias."""
    criticas = drift_periodo[drift_periodo["nivel"] == "Crítico"]["variable"].tolist()
    en_alerta = drift_periodo[drift_periodo["nivel"] == "Alerta"]["variable"].tolist()
    score = drift_periodo[drift_periodo["variable"] == "prob_no_pago"].iloc[0]

    mensajes = []
    if criticas:
        mensajes.append(("Crítico", f"Cambio fuerte en {len(criticas)} variable(s): {', '.join(criticas)}."))
    if en_alerta:
        mensajes.append(("Alerta", f"Cambio moderado en {len(en_alerta)} variable(s): {', '.join(en_alerta)}."))
    if score["nivel"] != "Estable":
        mensajes.append((score["nivel"], "Cambió la distribución de las probabilidades que da el modelo."))

    if score["nivel"] == "Crítico" or len(criticas) >= 3:
        mensajes.append(("Recomendación", "Reentrenar el modelo con datos recientes y revisar la calidad de la fuente de datos."))
    elif criticas or score["nivel"] == "Alerta":
        mensajes.append(("Recomendación", "Revisar las variables en alerta y medir el desempeño real del modelo con los pagos ya conocidos."))
    elif not mensajes:
        mensajes.append(("Estable", "No se detectaron cambios importantes. El modelo se puede seguir usando."))
    return mensajes


# =====================================================================
# 7. GRÁFICAS
# =====================================================================
def graficar_drift_en_el_tiempo(drift, ruta):
    """Mapa de colores: el PSI de cada variable en cada mes."""
    tabla = drift.pivot(index="variable", columns="periodo", values="psi")
    tabla = tabla.loc[tabla.max(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(12, 8))
    sns.heatmap(tabla, cmap="YlOrRd", vmin=0, vmax=0.5, annot=True, fmt=".2f", annot_kws={"size": 7},
                cbar_kws={"label": "PSI (más de 0,25 = cambio fuerte)"})
    plt.title("Evolución del drift (PSI) por variable y por mes")
    plt.xlabel("Mes")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(ruta, dpi=110)
    plt.close()


def graficar_historico_vs_actual(historico, actual, variable, ruta):
    """Compara la distribución de una variable: histórico vs mes actual."""
    plt.figure(figsize=(7, 4))
    if variable in VARIABLES_NUMERICAS + ["prob_no_pago"]:
        sns.kdeplot(historico[variable].dropna(), label="Histórico", fill=True, alpha=0.3, color="#2E86C1")
        sns.kdeplot(actual[variable].dropna(), label="Mes actual", fill=True, alpha=0.3, color="#C0392B")
    else:
        datos = pd.concat([historico.assign(grupo="Histórico"), actual.assign(grupo="Mes actual")])
        proporciones = (datos.groupby("grupo")[variable].value_counts(normalize=True).rename("proporción").reset_index())
        sns.barplot(data=proporciones, x=variable, y="proporción", hue="grupo")
    plt.title(f"{variable}: histórico vs mes actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta, dpi=110)
    plt.close()


# =====================================================================
# 8. FUNCIÓN PRINCIPAL
# =====================================================================
def main():
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    print("1. Construyendo la tabla de monitoreo...")
    tabla = construir_tabla_monitoreo()
    historico, periodos, meses_historico = dividir_historico_y_periodos(tabla)
    print(f"   Histórico: {meses_historico[0]} a {meses_historico[-1]} ({len(historico):,} registros)")

    print("2. Calculando el drift mes a mes...")
    drift = drift_en_el_tiempo(tabla)
    drift.round(4).to_csv(CARPETA_RESULTADOS / "monitoreo_drift.csv", index=False)
    graficar_drift_en_el_tiempo(drift, CARPETA_RESULTADOS / "drift_en_el_tiempo.png")

    resumen = (drift.groupby("periodo").agg(registros=("registros", "first"),
               criticas=("nivel", lambda s: int((s == "Crítico").sum())),
               alertas=("nivel", lambda s: int((s == "Alerta").sum()))))
    print("\n3. Variables en estado crítico / alerta por mes:")
    print(resumen.to_string())

    ultimo = drift["periodo"].max()
    print(f"\n4. Alertas del último mes analizado ({ultimo}):")
    for tipo, mensaje in generar_alertas(drift[drift["periodo"] == ultimo]):
        print(f"   [{tipo}] {mensaje}")

    peor = drift[drift["periodo"] == ultimo].sort_values("psi", ascending=False).iloc[0]["variable"]
    graficar_historico_vs_actual(historico, tabla[tabla["periodo"] == ultimo], peor,
                                 CARPETA_RESULTADOS / "historico_vs_actual.png")
    print(f"\nResultados guardados en: {CARPETA_RESULTADOS}")


if __name__ == "__main__":
    main()
