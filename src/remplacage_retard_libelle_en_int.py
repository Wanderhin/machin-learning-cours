"""
Script de transformation : conversion Oui/Non -> 1/0
------------------------------------------------------
Charge le fichier Excel, remplace les valeurs de la colonne
'retard_libelle' (Oui -> 1, Non -> 0), puis sauvegarde le résultat
dans un nouveau fichier Excel.
"""

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# 1. CHARGEMENT DU FICHIER
# ------------------------------------------------------------------
# Adaptez le chemin selon l'emplacement de CE script dans votre projet.
# Exemple de structure supposée :
#   projet/
#   ├── src/transformation.py   <- ce script
#   └── data/jeu_donnees_retards_projets_ML_1000.xlsx

base_dir = Path(__file__).parent.absolute()
chemin_fichier = base_dir.parent / "data" / "jeu_donnees_retards_projets_ML_1000.xlsx"

xls = pd.ExcelFile(chemin_fichier)
print("Feuilles disponibles :", xls.sheet_names)

NOM_FEUILLE = xls.sheet_names[1]  # <-- adapter si besoin
fich = pd.read_excel(chemin_fichier, sheet_name=NOM_FEUILLE)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("\nAvant transformation :")
print(fich["retard_libelle"].value_counts())

# ------------------------------------------------------------------
# 2. REMPLACEMENT Oui -> 1, Non -> 0
# ------------------------------------------------------------------
fich["retard_libelle"] = fich["retard_libelle"].replace({"Oui": 1, "Non": 0})

print("\nAprès transformation :")
print(fich["retard_libelle"].value_counts())
print(fich[["retard_libelle"]].head())

# ------------------------------------------------------------------
# 3. VÉRIFICATION DE COHÉRENCE (optionnel)
# ------------------------------------------------------------------
# Vérifie que retard_libelle correspond bien à la colonne retard
if "retard" in fich.columns:
    incoherences = (fich["retard_libelle"] != fich["retard"]).sum()
    print(f"\nNombre d'incohérences entre 'retard' et 'retard_libelle' : {incoherences}")

# ------------------------------------------------------------------
# 4. MISE À JOUR DU FICHIER ORIGINAL (écrase la feuille modifiée)
# ------------------------------------------------------------------
# ATTENTION : ceci écrase la feuille NOM_FEUILLE dans le fichier existant.
# Si le fichier contient d'autres feuilles (ex: une feuille "Description"),
# elles sont préservées grâce au mode "a" (append) + if_sheet_exists="replace".

with pd.ExcelWriter(chemin_fichier, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    fich.to_excel(writer, sheet_name=NOM_FEUILLE, index=False)

print(f"\nFichier mis à jour : {chemin_fichier} (feuille '{NOM_FEUILLE}')")