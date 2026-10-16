import joblib
import numpy as np
import pandas as pd

# 6. Rechargement du pipeline sérialisé
modele = joblib.load("modele_titanic.joblib")
print("Modèle rechargé :", modele)

# 7. Prédiction pour un nouveau passager
# Mêmes colonnes et même ordre qu'à l'entraînement (Sex : 0 = male, 1 = female)
nouveau_passager = pd.DataFrame([{
    "Pclass": 2,
    "Sex": 1,
    "Age": 22,
    "Fare": 28.31,
    "Embarked": 1
}])

prediction = modele.predict(nouveau_passager)
proba = modele.predict_proba(nouveau_passager)

print("Résultat :", "Survit" if prediction[0] == 1 else "Ne survit pas")
print(f"Probabilité de survie : {proba[0][1]:.2%}")
print("Après sérialisation :", prediction[0], proba[0])
