"""
Pruebas automáticas del proyecto (se ejecutan con:  pytest)
Sirven para que SonarCloud mida la cobertura y para comprobar que nada se rompió.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlops_pipeline"))

import pandas as pd

from ft_engineering import cargar_datos, crear_variables, limpiar_datos, preparar_datos
from model_monitoring import calcular_psi, clasificar_nivel


def test_cargar_datos_tiene_columna_objetivo():
    df = cargar_datos()
    assert "Pago_atiempo" in df.columns
    assert len(df) > 10000


def test_limpiar_datos_quita_edades_imposibles():
    df = cargar_datos().head(200).copy()
    df.loc[df.index[0], "edad_cliente"] = 5
    limpio = limpiar_datos(df)
    assert pd.isna(limpio.loc[df.index[0], "edad_cliente"])


def test_crear_variables_agrega_razones():
    df = crear_variables(limpiar_datos(cargar_datos().head(200)))
    for columna in ["cuota_sobre_salario", "mes_prestamo", "tiene_mora"]:
        assert columna in df.columns


def test_preparar_datos_separa_80_20():
    X_train, X_test, y_train, y_test, _ = preparar_datos()
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.2) < 0.01
    assert "puntaje" not in X_train.columns   # no se usa: causaría fuga de datos


def test_psi_de_distribuciones_iguales_es_cero():
    p = pd.Series([0.25, 0.25, 0.25, 0.25]).values
    assert calcular_psi(p, p) < 1e-6


def test_clasificar_nivel_critico():
    assert clasificar_nivel(0.5) == "Crítico"
    assert clasificar_nivel(0.01) == "Estable"
