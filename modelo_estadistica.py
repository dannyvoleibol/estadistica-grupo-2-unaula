# Reproducción del modelo del Grupo 2 (requiere pandas, scikit-learn)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

df = pd.read_excel("High_Accuracy_Sport_Injury_Dataset.xlsx")
X = df.drop(columns=["Injury_Risk"])
y = df["Injury_Risk"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print("Matriz",confusion_matrix(y_test,pred,labels=[0,1]))
print("Exactitud",accuracy_score(y_test,pred))
print("Sensibilidad",recall_score(y_test,pred,pos_label=1))
print("Precisión",precision_score(y_test,pred,pos_label=1))
print("Referencia mayoritaria",(y_test==y_train.mode().iloc[0]).mean())

# Pre/post: emparejamiento dentro de cada fila por atleta_id
pre = pd.read_csv("base_prepost_entrenamiento.csv")
for a,b in [("cmj_pre_cm","cmj_post_cm"),("sprint30_pre_s","sprint30_post_s"),("vo2max_pre","vo2max_post")]:
    pre["cambio_"+a] = pre[b]-pre[a]
    print(a,pre.groupby("grupo")[[a,b,"cambio_"+a]].mean())