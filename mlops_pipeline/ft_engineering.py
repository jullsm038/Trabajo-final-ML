"""
ft_engineering.py
Prepara los datos para entrenar los modelos.

ÍNDICE
  1. Datos fijos: rutas y listas de columnas
  2. Cargar los datos
  3. Limpiar los datos
  4. Crear variables nuevas
  5. Pipeline de transformación
  6. Función principal: preparar_datos
  7. Prueba rápida

Uso desde otro archivo:
    from ft_engineering import preparar_datos
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos()
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# =====================================================================
# 1. DATOS FIJOS
# Rutas, nombre de lo que se quiere predecir y listas de columnas.
# =====================================================================
RUTA_DATOS = Path(__file__).resolve().parent.parent / "Base_de_datos.csv"
OBJETIVO = "Pago_atiempo"   # lo que queremos predecir
SEMILLA = 42                # para que los resultados siempre sean los mismos

VALORES_NULOS = ["", " ", "NA", "N/A", "na", "null", "NULL", "None", "nan", "NaN", "-"]
TENDENCIAS_VALIDAS = ["Creciente", "Estable", "Decreciente"]

# Dinero: tienen cola larga, por eso se les aplica logaritmo
COLUMNAS_DINERO = ["capital_prestado", "salario_cliente", "total_otros_prestamos",
                   "cuota_pactada", "saldo_mora", "saldo_total",
                   "promedio_ingresos_datacredito"]

# Variables nuevas que comparan el crédito con el salario
COLUMNAS_RAZONES = ["cuota_sobre_salario", "capital_sobre_salario",
                    "otros_prestamos_sobre_salario"]

# Otros números: solo se rellenan los vacíos y se llevan a la misma escala
COLUMNAS_NUMERICAS_SIMPLES = ["plazo_meses", "edad_cliente", "puntaje_datacredito",
                              "cant_creditosvigentes", "huella_consulta",
                              "creditos_sectorFinanciero", "creditos_sectorCooperativo",
                              "creditos_sectorReal", "mes_prestamo", "tiene_mora"]

# Categorías: se convierten en columnas de 0 y 1
COLUMNAS_CATEGORICAS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]
# No se usan: saldo_principal (casi igual a saldo_total) y puntaje (posible fuga de datos)


# =====================================================================
# 2. CARGAR LOS DATOS
# Lee el CSV y escribe todos los vacíos de la misma forma.
# =====================================================================
def cargar_datos(ruta=RUTA_DATOS):
    df = pd.read_csv(ruta)
    return df.replace(VALORES_NULOS, np.nan)


# =====================================================================
# 3. LIMPIAR LOS DATOS
# Aplica las reglas de validación del EDA (parte 5.1). Usa una copia.
# =====================================================================
def limpiar_datos(df):
    df = df.copy()

    # tendencia_ingresos: solo 3 categorías válidas; lo demás es "Sin información"
    invalidos = df["tendencia_ingresos"].notna() & ~df["tendencia_ingresos"].isin(TENDENCIAS_VALIDAS)
    df.loc[invalidos, "tendencia_ingresos"] = np.nan
    df["tendencia_ingresos"] = df["tendencia_ingresos"].fillna("Sin información")

    # Valores imposibles pasan a vacío (después se rellenan)
    df.loc[(df["edad_cliente"] < 18) | (df["edad_cliente"] > 90), "edad_cliente"] = np.nan
    df.loc[(df["salario_cliente"] <= 0) | (df["salario_cliente"] > 5e7), "salario_cliente"] = np.nan
    df.loc[df["puntaje_datacredito"] <= 0, "puntaje_datacredito"] = np.nan
    df.loc[df["plazo_meses"] > 48, "plazo_meses"] = np.nan
    df.loc[df["total_otros_prestamos"] > 1e9, "total_otros_prestamos"] = np.nan

    # tipo_credito: los tipos muy raros se agrupan como "Otros"
    tipo = df["tipo_credito"].astype(str)
    df["tipo_credito"] = tipo.where(tipo.isin(["4", "9"]), "Otros")
    return df


# =====================================================================
# 4. CREAR VARIABLES NUEVAS
# Columnas nuevas propuestas en el EDA (parte 4.3).
# =====================================================================
def crear_variables(df):
    df = df.copy()
    salario = df["salario_cliente"].where(df["salario_cliente"] > 0)
    df["cuota_sobre_salario"] = df["cuota_pactada"] / salario
    df["capital_sobre_salario"] = df["capital_prestado"] / salario
    df["otros_prestamos_sobre_salario"] = df["total_otros_prestamos"] / salario
    df["mes_prestamo"] = pd.to_datetime(df["fecha_prestamo"]).dt.month
    df["tiene_mora"] = (df["saldo_mora"].fillna(0) > 0).astype(int)
    return df


# =====================================================================
# 5. PIPELINE DE TRANSFORMACIÓN
# Cadena de pasos que se aplica siempre igual. Tiene tres ramas:
#   dinero      -> rellenar vacíos, logaritmo, misma escala
#   numéricas   -> rellenar vacíos, misma escala
#   categorías  -> rellenar vacíos, columnas 0/1
# =====================================================================
def _log(x):
    """Logaritmo que funciona con el cero."""
    return np.log1p(np.clip(x, 0, None))


def construir_preprocesador(columnas_dinero, columnas_razones, columnas_simples, columnas_categoricas):
    rama_dinero = Pipeline([
        ("rellenar", SimpleImputer(strategy="median")),
        ("logaritmo", FunctionTransformer(_log, feature_names_out="one-to-one")),
        ("escalar", StandardScaler()),
    ])
    rama_numeros = Pipeline([
        ("rellenar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    rama_categorias = Pipeline([
        ("rellenar", SimpleImputer(strategy="most_frequent")),
        ("columnas_0_1", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("dinero", rama_dinero, columnas_dinero + columnas_razones),
        ("numericas", rama_numeros, columnas_simples),
        ("categoricas", rama_categorias, columnas_categoricas),
    ])


# =====================================================================
# 6. FUNCIÓN PRINCIPAL
# Carga, limpia, crea variables y divide en entrenamiento y evaluación.
# usar_puntaje=False deja por fuera `puntaje` (posible fuga de datos).
# =====================================================================
def preparar_datos(usar_puntaje=False, tamano_prueba=0.2):
    df = crear_variables(limpiar_datos(cargar_datos()))
    simples = list(COLUMNAS_NUMERICAS_SIMPLES)
    if usar_puntaje:
        simples.append("puntaje")

    columnas = COLUMNAS_DINERO + COLUMNAS_RAZONES + simples + COLUMNAS_CATEGORICAS
    X = df[columnas]
    y = df[OBJETIVO]

    # 80 % para entrenar y 20 % para evaluar, con la misma proporción de morosos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tamano_prueba, stratify=y, random_state=SEMILLA)

    preprocesador = construir_preprocesador(
        COLUMNAS_DINERO, COLUMNAS_RAZONES, simples, COLUMNAS_CATEGORICAS)
    return X_train, X_test, y_train, y_test, preprocesador


# =====================================================================
# 7. PRUEBA RÁPIDA
# Solo corre cuando se ejecuta este archivo directamente.
# =====================================================================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, prep = preparar_datos()
    print("Entrenamiento:", X_train.shape, "| Evaluación:", X_test.shape)
    X_listo = prep.fit_transform(X_train)
    print("Columnas después de transformar:", X_listo.shape[1])
    print("Vacíos que quedan:", int(np.isnan(X_listo).sum()))
