"""
Script d'analyse des valeurs manquantes
----------------------------------------
Charge un fichier Excel et identifie précisément où se trouvent
les valeurs manquantes (colonnes, pourcentages, lignes concernées).
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 1. CHARGEMENT DU FICHIER
# ------------------------------------------------------------------
# Adaptez le chemin selon l'emplacement de CE script dans votre projet.
# Exemple de structure supposée :
#   projet/
#   ├── src/analyse_valeurs_manquantes.py   <- ce script
#   └── data/jeu_donnees_retards_projets_ML_1000.xlsx

base_dir = Path(__file__).parent.absolute()
chemin_fichier = base_dir.parent / "data" / "jeu_donnees_retards_projets_ML_1000.xlsx"

# Afficher les feuilles disponibles dans le fichier Excel
xls = pd.ExcelFile(chemin_fichier)
print("Feuilles disponibles :", xls.sheet_names)

# Adapter l'index/nom de la feuille contenant les vraies données
NOM_FEUILLE = xls.sheet_names[1]  # <-- modifier si besoin
fich = pd.read_excel(chemin_fichier, sheet_name=NOM_FEUILLE)

# Options d'affichage pour ne rien tronquer
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("\nAperçu des données :")
print(fich.head())
print("\nDimensions :", fich.shape)

# ------------------------------------------------------------------
# 2. NOMBRE DE VALEURS MANQUANTES PAR COLONNE
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("NOMBRE DE VALEURS MANQUANTES PAR COLONNE")
print("=" * 60)
manquants = fich.isnull().sum()
print(manquants)

# ------------------------------------------------------------------
# 3. RÉSUMÉ (NOMBRE + POURCENTAGE), UNIQUEMENT LES COLONNES CONCERNÉES
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("RÉSUMÉ DES COLONNES AVEC VALEURS MANQUANTES")
print("=" * 60)

resume_manquants = pd.DataFrame({
    'Nombre_manquant': fich.isnull().sum(),
    'Pourcentage_%': (fich.isnull().sum() / len(fich) * 100).round(2)
})
resume_manquants = resume_manquants[resume_manquants['Nombre_manquant'] > 0]
resume_manquants = resume_manquants.sort_values('Nombre_manquant', ascending=False)

if resume_manquants.empty:
    print("Aucune valeur manquante détectée dans le jeu de données.")
else:
    print(resume_manquants)

# ------------------------------------------------------------------
# 4. TOTAL GLOBAL
# ------------------------------------------------------------------
total_manquants = fich.isnull().sum().sum()
print(f"\nTotal de valeurs manquantes dans tout le fichier : {total_manquants}")

# ------------------------------------------------------------------
# 5. LIGNES CONTENANT AU MOINS UNE VALEUR MANQUANTE
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("LIGNES CONTENANT AU MOINS UNE VALEUR MANQUANTE")
print("=" * 60)
lignes_avec_manquants = fich[fich.isnull().any(axis=1)]
print(f"Nombre de lignes concernées : {lignes_avec_manquants.shape[0]}")
print(lignes_avec_manquants)

# ------------------------------------------------------------------
# 6. LOCALISATION PRÉCISE (LIGNE, COLONNE) DE CHAQUE VALEUR MANQUANTE
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("LOCALISATION EXACTE (index de ligne, colonne)")
print("=" * 60)

positions = [
    (ligne, colonne)
    for colonne in fich.columns
    for ligne in fich.index[fich[colonne].isnull()]
]

if positions:
    for ligne, colonne in positions:
        print(f"  -> Ligne {ligne} | Colonne '{colonne}'")
else:
    print("Aucune position manquante à afficher.")

# ------------------------------------------------------------------
# 7. VISUALISATION (carte des valeurs manquantes)
# ------------------------------------------------------------------
if total_manquants > 0:
    plt.figure(figsize=(10, 6))
    sns.heatmap(fich.isnull(), cbar=False, cmap='viridis')
    plt.title("Carte des valeurs manquantes")
    plt.tight_layout()
    plt.show()
else:
    print("\nPas de visualisation nécessaire : aucune valeur manquante.")