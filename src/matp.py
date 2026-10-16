from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# chargement de donnee
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

plt.plot(x, y)

# charger les donnnees et les diviser en 80 pour l'entrainement et 20 pour le test
base_dir = Path(__file__).parent.absolute()
FICHIER = base_dir.parent / "data" / "jeu_donnees_retards_projets_ML_1000.xlsx"
df = pd.read_excel(FICHIER, sheet_name="Donnees")
#train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
#print("train set ", len(train_set), " test set ", len(test_set))


# Variables explicatives et cible
X = df.drop(columns=["id_projet", "retard", "retard_libelle", "retard_jours"])
y = df["retard"]

# Séparation 80 % / 20 % (stratifiée : même proportion de retards partout)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modèle
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

# Évaluation sur le test_set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred))
print("AUC      :", roc_auc_score(y_test, y_proba))
print("##############################################")
print(confusion_matrix(y_test, y_pred))
print("##############################################")
print(classification_report(y_test, y_pred))
print("##############################################")
# Importance des variables
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importances.plot(kind="barh")
plt.title("Importance des variables")
plt.tight_layout()
plt.savefig("importance_variables3.png")
plt.show()