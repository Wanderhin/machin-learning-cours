"""Chargement des 5 millions de lignes (5 fichiers Excel) avec cache.

1re exécution : lecture des 5 fichiers Excel (quelques minutes), puis création
d'un cache 'cache_5M.pkl'. Les exécutions suivantes lisent le cache (quelques secondes).
Pour lire Excel vite : pip install python-calamine  (sinon openpyxl, beaucoup plus lent).
"""
from pathlib import Path

import pandas as pd

FICHIERS = [f"data/donnees_impayes_partie{i}_sur_5.xlsx" for i in range(1, 6)]
CACHE = Path("cache_5M.pkl")
COLONNES = ["retards_12m", "montant_facture", "anciennete_mois", "impaye"]  # id_client inutile pour le ML


def _lire_un_fichier(chemin: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(chemin, sheet_name="Donnees", usecols=COLONNES, engine="calamine")
        print("  (lecture rapide avec python-calamine)")
    except (ImportError, ValueError):
        print("  (python-calamine absent : lecture avec openpyxl, c'est lent)")
        df = pd.read_excel(chemin, sheet_name="Donnees", usecols=COLONNES)
    # types compacts pour économiser la mémoire (float32 gère aussi les valeurs manquantes)
    for c in ["retards_12m", "montant_facture", "anciennete_mois"]:
        df[c] = df[c].astype("float32")
    df["impaye"] = df["impaye"].astype("int8")
    return df


def charger_donnees(nb_lignes_max=None, graine=42, forcer=False) -> pd.DataFrame:
    """Retourne le jeu complet. nb_lignes_max : échantillon aléatoire pour tester vite."""
    if CACHE.exists() and not forcer:
        print(f"Lecture du cache {CACHE} ...")
        df = pd.read_pickle(CACHE)
    else:
        morceaux = []
        for f in FICHIERS:
            print(f"Lecture de {f} ...")
            morceaux.append(_lire_un_fichier(f))
        df = pd.concat(morceaux, ignore_index=True)
        df.to_pickle(CACHE)
        print(f"Cache enregistré : {CACHE}")

    if nb_lignes_max and len(df) > nb_lignes_max:
        df = df.sample(nb_lignes_max, random_state=graine).reset_index(drop=True)
    print(f"{len(df):,} lignes chargées | taux d'impayés = {df['impaye'].mean():.3f}")
    return df