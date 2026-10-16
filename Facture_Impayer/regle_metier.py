"""Règles métiers expertes : grille de points, score, classe de risque, règles de priorité.

Utilisé à l'entraînement (V2) ET au déploiement : même code = mêmes résultats.
Les seuils sont des propositions, à valider avec le métier. Modifiables ici.
"""
import numpy as np
import pandas as pd

# ---- Seuils ----
MONTANT_MOYEN = 100_000      # FCFA
MONTANT_ELEVE = 250_000      # FCFA
ANCIENNETE_1_AN = 12         # mois
ANCIENNETE_2_ANS = 24        # mois
ANCIENNETE_5_ANS = 60        # mois

COLONNES_BRUTES = ["retards_12m", "montant_facture", "anciennete_mois"]

# V2 : les 3 variables brutes + les variables issues des règles
FEATURES_ML = COLONNES_BRUTES + [
    "pts_retards", "pts_montant", "pts_anciennete", "score_regles", "rp1",
]

CLASSES = ["Faible", "Moyen", "Élevé", "Traitement manuel"]

DECISIONS = {
    "Faible": "Facturation normale",
    "Moyen": "Rappel avant échéance, suivi renforcé",
    "Élevé": "Paiement anticipé ou garantie, validation par un gestionnaire",
    "Traitement manuel": "Traitement manuel (données manquantes ou hors plage)",
}


def appliquer_regles(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute points, score, classe de risque et drapeaux RP1 / RP2 (hors_plage)."""
    out = df.copy()
    for c in COLONNES_BRUTES:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    r, m, a = out["retards_12m"], out["montant_facture"], out["anciennete_mois"]

    # RP2 : valeur manquante ou hors plage -> traitement manuel
    out["hors_plage"] = (
        r.isna() | m.isna() | a.isna() | (r < 0) | (r > 12) | (m <= 0) | (a < 0)
    )

    # 1) Grille de points
    out["pts_retards"] = np.select([r == 0, r <= 2, r == 3], [0, 1, 3], default=5).astype("int8")
    out["pts_montant"] = np.select([m < MONTANT_MOYEN, m <= MONTANT_ELEVE], [0, 1], default=2).astype("int8")
    out["pts_anciennete"] = np.select([a > ANCIENNETE_5_ANS, a >= ANCIENNETE_2_ANS], [0, 1], default=2).astype("int8")

    # 2) Score total (0 à 9)
    out["score_regles"] = (out["pts_retards"] + out["pts_montant"] + out["pts_anciennete"]).astype("int8")

    # RP1 : client de moins d'1 an avec une facture > 250 000 FCFA -> risque élevé
    out["rp1"] = ((a < ANCIENNETE_1_AN) & (m > MONTANT_ELEVE) & ~out["hors_plage"]).astype("int8")

    # 3) Classe de risque, avec priorités : RP2 > RP1 > score
    code = np.select(
        [out["hors_plage"], out["rp1"] == 1, out["score_regles"] >= 5, out["score_regles"] >= 3],
        [3, 2, 2, 1],
        default=0,
    )
    out["classe_regles"] = pd.Categorical.from_codes(code, categories=CLASSES)
    return out