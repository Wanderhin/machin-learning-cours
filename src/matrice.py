import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

base_dir = Path(__file__).parent.absolute()
chemin_fichier = base_dir.parent / "data" / "jeu_donnees_retards_projets_ML_1000.xlsx"

xls = pd.ExcelFile(chemin_fichier)
fich = pd.read_excel(chemin_fichier, sheet_name=xls.sheet_names[1])

print(fich.columns.tolist())

# y = variable cible, ciblée par son nom
y = fich["retard"]

# X = toutes les colonnes sauf la cible
X = fich.drop(columns=["retard"])

print("Variable y :")
print(y.value_counts())

print("Variables X :")
print(X.head())
print(X.shape)