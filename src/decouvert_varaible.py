import pandas as pd
from pathlib import Path

base_dir = Path(__file__).parent.absolute()
chemin_fichier = base_dir.parent / "data" / "jeu_donnees_retards_projets_ML_1000.xlsx"

xls = pd.ExcelFile(chemin_fichier)
print(xls.sheet_names)

fich = pd.read_excel(chemin_fichier, sheet_name="Donnees")
print(fich.head())
print(fich.shape)
print(fich.columns.tolist())

# Variable cible
y = fich["retard"]

# Variables explicatives (tout sauf la cible)
X = fich.drop(columns=["retard"])

print("X shape :", X.shape)
print("y shape :", y.shape)
print(X.dtypes)
print(y.value_counts())  # vérifier l'équilibre des classes (0 vs 1)