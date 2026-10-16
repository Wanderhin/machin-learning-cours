from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             RocCurveDisplay)

base_dir = Path(__file__).parent.absolute()
FICHIER = base_dir.parent / "data" / "jeu_donnees_retards_projets_ML_1000.xlsx"
df = pd.read_excel(FICHIER, sheet_name="Donnees")


# Variables explicatives et cible
X = df.drop(columns=["id_projet", "retard", "retard_libelle", "retard_jours"])
y = df["retard"]

# Séparation 80 % / 20 % (stratifiée : même proportion de retards partout)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Modèle : on normalise les variables puis on applique la régression logistique
model_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)
model_lr.fit(X_train, y_train)

# Évaluation sur le test_set
y_pred = model_lr.predict(X_test)
y_proba = model_lr.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred))
print("AUC      :", roc_auc_score(y_test, y_proba))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Coefficients : influence de chaque variable
coefs = pd.Series(
    model_lr.named_steps["logisticregression"].coef_[0],
    index=X.columns
).sort_values()
print(coefs)

coefs.plot(kind="barh")
plt.title("Coefficients de la régression logistique")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()

# Courbe ROC
RocCurveDisplay.from_estimator(model_lr, X_test, y_test)
plt.show()
