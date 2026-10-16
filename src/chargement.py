import pandas as pd
from pathlib import Path
base_dir = Path.cwd()
fich = pd.read_excel(base_dir.parent / "data" / "jeu_donnees_retards_projets_ML_1000.xlsx")
print(fich.head())
print("###########################################################")
print(fich.shape)
print("###########################################################")

print(fich.describe())
print("###########################################################")

print(fich.isnull().sum())
print("###########################################################")
