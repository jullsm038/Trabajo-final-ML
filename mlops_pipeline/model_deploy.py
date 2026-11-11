"""
model_deploy.py
API para usar el mejor modelo (regresión logística) desde cualquier programa.

ÍNDICE
  1. Datos fijos
  2. Formato de los datos de entrada
  3. Función que predice
  4. La API (rutas)

Cómo encenderla (desde la carpeta mlops_pipeline):
    uvicorn model_deploy:app --reload --port 8000
Luego abre en el navegador:  http://127.0.0.1:8000/docs
"""
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ft_engineering import (COLUMNAS_CATEGORICAS, COLUMNAS_DINERO, COLUMNAS_NUMERICAS_SIMPLES,
                            COLUMNAS_RAZONES, crear_variables, limpiar_datos)


# =====================================================================
# 1. DATOS FIJOS
# =====================================================================
RUTA_MODELO = Path(__file__).resolve().parent / "mejor_modelo.joblib"
COLUMNAS_MODELO = COLUMNAS_DINERO + COLUMNAS_RAZONES + COLUMNAS_NUMERICAS_SIMPLES + COLUMNAS_CATEGORICAS
UMBRAL = 0.5   # si la probabilidad de NO pagar es mayor a esto, se avisa "no pagará"

# Columnas originales que necesita el modelo (las demás se crean solas)
COLUMNAS_ENTRADA = [
    "tipo_credito", "fecha_prestamo", "capital_prestado", "plazo_meses", "edad_cliente",
    "tipo_laboral", "salario_cliente", "total_otros_prestamos", "cuota_pactada",
    "puntaje_datacredito", "cant_creditosvigentes", "huella_consulta", "saldo_mora",
    "saldo_total", "creditos_sectorFinanciero", "creditos_sectorCooperativo",
    "creditos_sectorReal", "promedio_ingresos_datacredito", "tendencia_ingresos",
]

modelo = joblib.load(RUTA_MODELO)   # se carga una sola vez, al encender la API


# =====================================================================
# 2. FORMATO DE LOS DATOS DE ENTRADA
# Cada campo es un dato del cliente. Los opcionales pueden venir vacíos.
# =====================================================================
class Credito(BaseModel):
    tipo_credito: str
    fecha_prestamo: str
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float = 0
    cuota_pactada: float
    puntaje_datacredito: Optional[float] = None
    cant_creditosvigentes: int = 0
    huella_consulta: int = 0
    saldo_mora: float = 0
    saldo_total: float = 0
    creditos_sectorFinanciero: int = 0
    creditos_sectorCooperativo: int = 0
    creditos_sectorReal: int = 0
    promedio_ingresos_datacredito: Optional[float] = None
    tendencia_ingresos: Optional[str] = None


# =====================================================================
# 3. FUNCIÓN QUE PREDICE
# Aplica la misma limpieza que en el entrenamiento y llama al modelo.
# =====================================================================
def predecir(registros: List[dict]) -> List[dict]:
    df = pd.DataFrame(registros)
    for columna in COLUMNAS_ENTRADA:            # si falta alguna, queda vacía
        if columna not in df.columns:
            df[columna] = None
    df = crear_variables(limpiar_datos(df[COLUMNAS_ENTRADA]))[COLUMNAS_MODELO]

    columna = list(modelo.classes_).index(0)
    prob_no_pago = modelo.predict_proba(df)[:, columna]
    resultados = []
    for p in prob_no_pago:
        resultados.append({
            "prob_no_pago": round(float(p), 4),
            "prediccion": "No pagará a tiempo" if p > UMBRAL else "Pagará a tiempo",
        })
    return resultados


# =====================================================================
# 4. LA API
# =====================================================================
app = FastAPI(title="API de riesgo de crédito",
              description="Predice si un cliente pagará su crédito a tiempo.")


@app.get("/")
def inicio():
    return {"mensaje": "API activa. Entra a /docs para probarla."}


@app.post("/predict")
def predict(credito: Credito):
    """Predice un solo crédito."""
    return predecir([credito.model_dump()])[0]


@app.post("/predict_batch")
def predict_batch(creditos: List[Credito]):
    """Predice varios créditos a la vez (por lotes)."""
    return predecir([c.model_dump() for c in creditos])
