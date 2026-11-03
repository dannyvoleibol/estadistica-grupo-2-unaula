
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    recall_score, precision_score
)

st.set_page_config(
    page_title="Estadistica Deportiva UNAULA",
    layout="wide"
)

st.title("Analisis Estadistico Deportivo")
st.subheader("Trabajo final - Grupo 2 - UNAULA")
st.write("Jose Daniel Chaves Arevalo")
st.info("Analisis academico. No sustituye una evaluacion clinica.")

df = pd.read_excel("High_Accuracy_Sport_Injury_Dataset.xlsx")

st.header("1. Base de datos")
st.write("Numero de registros:", len(df))
st.dataframe(df.head(10))

X = df.drop(columns=["Injury_Risk"])
y = df["Injury_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

modelo = RandomForestClassifier(
    n_estimators=200, random_state=42
)
modelo.fit(X_train, y_train)
pred = modelo.predict(X_test)

st.header("2. Modelo Random Forest")

c1, c2, c3 = st.columns(3)
c1.metric("Exactitud", f"{accuracy_score(y_test, pred):.2%}")
c2.metric("Sensibilidad", f"{recall_score(y_test, pred):.2%}")
c3.metric("Precision", f"{precision_score(y_test, pred):.2%}")

matriz = confusion_matrix(y_test, pred, labels=[0, 1])

st.subheader("Matriz de confusion")
st.dataframe(pd.DataFrame(
    matriz,
    index=["Real 0", "Real 1"],
    columns=["Predicho 0", "Predicho 1"]
))

st.write("Falsos positivos:", int(matriz[0, 1]))
st.write("Falsos negativos:", int(matriz[1, 0]))

st.header("3. Resultados antes y despues")

pre = pd.read_csv("base_prepost_entrenamiento.csv")

pruebas = [
    ("Salto CMJ (cm)", "cmj_pre_cm", "cmj_post_cm"),
    ("Sprint 30 metros (s)", "sprint30_pre_s", "sprint30_post_s"),
    ("VO2max", "vo2max_pre", "vo2max_post")
]

for nombre, antes, despues in pruebas:
    st.subheader(nombre)
    resumen = pre.groupby("grupo")[[antes, despues]].mean()
    resumen["Cambio"] = resumen[despues] - resumen[antes]
    st.dataframe(resumen.round(3))
    st.bar_chart(resumen[[antes, despues]])

st.success("Analisis completado correctamente.")
