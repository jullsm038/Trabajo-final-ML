"""
model_training_evaluation.py
Entrena varios modelos, los compara y guarda el mejor.

ÍNDICE
  1. Herramientas y datos fijos
  2. Métricas: summarize_classification
  3. Armar un modelo: build_model
  4. Lista de modelos a comparar
  5. Entrenar y evaluar
  6. Elegir el mejor modelo
  7. Gráficas comparativas
  8. Prueba con la variable puntaje (fuga de datos)
  9. Función principal

Lo importante es detectar a los clientes que NO pagan (solo el 4,7 %).
Por eso todas las métricas miran la clase 0 ("no pagó a tiempo"):
  - recall:    de los que no pagan, cuántos detecta el modelo
  - precisión: de los que el modelo marca como "no pagará", cuántos de verdad no pagan
  - f1:        mezcla de las dos anteriores
  - roc_auc:   qué tan bien ordena a los clientes de más a menos riesgosos

Se ejecuta con:  python mlops_pipeline/model_training_evaluation.py
"""
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # guarda las gráficas como imagen, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ft_engineering import SEMILLA, preparar_datos


# =====================================================================
# 1. HERRAMIENTAS Y DATOS FIJOS
# =====================================================================
warnings.filterwarnings("ignore")
CARPETA = Path(__file__).resolve().parent
CARPETA_RESULTADOS = CARPETA / "resultados"
CLASE_MALA = 0  # "no pagó a tiempo"


# =====================================================================
# 2. MÉTRICAS
# Calcula las métricas de un modelo mirando la clase "no paga".
# =====================================================================
def summarize_classification(y_true, y_pred, y_proba_mala):
    y_mala = (np.asarray(y_true) == CLASE_MALA).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=CLASE_MALA, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=CLASE_MALA),
        "f1": f1_score(y_true, y_pred, pos_label=CLASE_MALA),
        "roc_auc": roc_auc_score(y_mala, y_proba_mala),
        "pr_auc": average_precision_score(y_mala, y_proba_mala),
    }


# =====================================================================
# 3. ARMAR UN MODELO
# Une el preprocesador (de ft_engineering) con el modelo en un pipeline.
# =====================================================================
def build_model(estimador, preprocesador):
    return Pipeline([("preprocesador", preprocesador), ("modelo", estimador)])


def probabilidad_mala(modelo, X):
    """Probabilidad de que el cliente NO pague."""
    columna = list(modelo.classes_).index(CLASE_MALA)
    return modelo.predict_proba(X)[:, columna]


# =====================================================================
# 4. LISTA DE MODELOS A COMPARAR
# class_weight="balanced" le da más peso a la clase que casi no aparece.
# =====================================================================
def definir_modelos():
    return {
        "Regresión logística": LogisticRegression(max_iter=1000, class_weight="balanced",
                                                  random_state=SEMILLA),
        "Árbol de decisión": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                    class_weight="balanced", random_state=SEMILLA),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                class_weight="balanced", n_jobs=-1,
                                                random_state=SEMILLA),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEMILLA),
    }


# =====================================================================
# 5. ENTRENAR Y EVALUAR
# Entrena cada modelo, lo prueba con datos que no vio y mide su
# consistencia repitiendo la prueba 5 veces con partes distintas.
# =====================================================================
def entrenar_y_evaluar(X_train, X_test, y_train, y_test, preprocesador):
    filas, entrenados, probas = [], {}, {}
    validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    for nombre, estimador in definir_modelos().items():
        modelo = build_model(estimador, preprocesador)

        inicio = time.time()
        modelo.fit(X_train, y_train)
        segundos = time.time() - inicio

        y_pred = modelo.predict(X_test)
        proba = probabilidad_mala(modelo, X_test)
        fila = {"modelo": nombre, **summarize_classification(y_test, y_pred, proba)}

        cv = cross_validate(modelo, X_train, y_train, cv=validacion, n_jobs=1,
                            scoring={"auc": "roc_auc"})
        fila["cv_auc_promedio"] = cv["test_auc"].mean()
        fila["cv_auc_variacion"] = cv["test_auc"].std()
        fila["segundos_entrenamiento"] = segundos
        filas.append(fila)
        entrenados[nombre] = modelo
        probas[nombre] = proba
        print(f"  listo: {nombre}")

    tabla = pd.DataFrame(filas).set_index("modelo")
    return tabla, entrenados, probas


# =====================================================================
# 6. ELEGIR EL MEJOR MODELO
# Se mira: rendimiento (f1 y AUC), consistencia (poca variación)
# y escalabilidad (que entrene rápido).
# =====================================================================
def elegir_mejor(tabla):
    puntaje = tabla["f1"].rank() + tabla["roc_auc"].rank()
    puntaje += (-tabla["cv_auc_variacion"]).rank() * 0.5
    puntaje += (-tabla["segundos_entrenamiento"]).rank() * 0.5
    return puntaje.idxmax()


# =====================================================================
# 7. GRÁFICAS COMPARATIVAS
# Se guardan en la carpeta "resultados".
# =====================================================================
def graficar_comparacion(tabla, probas, y_test, mejor, entrenados, X_test):
    CARPETA_RESULTADOS.mkdir(exist_ok=True)

    # 7.1 Barras con las métricas principales
    metricas = ["precision", "recall", "f1", "roc_auc"]
    largo = tabla[metricas].reset_index().melt(id_vars="modelo", var_name="métrica",
                                               value_name="valor")
    plt.figure(figsize=(11, 5))
    sns.barplot(data=largo, x="métrica", y="valor", hue="modelo")
    plt.title("Comparación de modelos (clase: no paga a tiempo)")
    plt.ylim(0, 1)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(CARPETA_RESULTADOS / "comparacion_metricas.png", dpi=120)
    plt.close()

    # 7.2 Curvas ROC
    y_mala = (np.asarray(y_test) == CLASE_MALA).astype(int)
    plt.figure(figsize=(7, 6))
    for nombre, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_mala, proba)
        plt.plot(fpr, tpr, label=f"{nombre} (AUC {tabla.loc[nombre, 'roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Al azar")
    plt.xlabel("Falsos positivos (buenos marcados como malos)")
    plt.ylabel("Verdaderos positivos (malos detectados)")
    plt.title("Curvas ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CARPETA_RESULTADOS / "curvas_roc.png", dpi=120)
    plt.close()

    # 7.3 Matriz de confusión del mejor modelo
    y_pred = entrenados[mejor].predict(X_test)
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No paga", "Paga"], yticklabels=["No paga", "Paga"])
    plt.xlabel("Lo que dijo el modelo")
    plt.ylabel("Lo que pasó en realidad")
    plt.title(f"Matriz de confusión: {mejor}")
    plt.tight_layout()
    plt.savefig(CARPETA_RESULTADOS / "matriz_confusion_mejor.png", dpi=120)
    plt.close()


# =====================================================================
# 8. PRUEBA CON LA VARIABLE PUNTAJE
# Muestra qué pasa si se usa `puntaje` (resultado perfecto = fuga de datos).
# =====================================================================
def comparar_con_puntaje():
    X_tr, X_te, y_tr, y_te, prep = preparar_datos(usar_puntaje=True)
    modelo = build_model(RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                                class_weight="balanced", n_jobs=-1,
                                                random_state=SEMILLA), prep)
    modelo.fit(X_tr, y_tr)
    return summarize_classification(y_te, modelo.predict(X_te), probabilidad_mala(modelo, X_te))


# =====================================================================
# 9. FUNCIÓN PRINCIPAL
# Corre todo en orden y guarda tabla, gráficas y el mejor modelo.
# =====================================================================
def main():
    print("1. Preparando datos...")
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos(usar_puntaje=False)
    print(f"   Entrenamiento: {X_train.shape[0]} | Evaluación: {X_test.shape[0]}")

    print("2. Entrenando modelos...")
    tabla, entrenados, probas = entrenar_y_evaluar(X_train, X_test, y_train, y_test, preprocesador)

    mejor = elegir_mejor(tabla)
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    tabla.round(3).to_csv(CARPETA_RESULTADOS / "tabla_resumen.csv")
    graficar_comparacion(tabla, probas, y_test, mejor, entrenados, X_test)
    joblib.dump(entrenados[mejor], CARPETA / "mejor_modelo.joblib")

    print("\n3. TABLA RESUMEN (sin la variable puntaje)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tabla.round(3))
    print(f"\nMejor modelo: {mejor}")

    print("\n4. Prueba con `puntaje` (posible fuga de datos), Random Forest:")
    con = comparar_con_puntaje()
    print({k: round(v, 3) for k, v in con.items()})
    print("\nResultados guardados en:", CARPETA_RESULTADOS)


if __name__ == "__main__":
    main()
    
