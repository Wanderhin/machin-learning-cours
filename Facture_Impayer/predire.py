"""Utilisation d'un modèle sauvegardé (V1 ou V2) : base du futur service déployé."""
from functools import lru_cache

import joblib
import pandas as pd

from Facture_Impayer.regle_metier import appliquer_regles, DECISIONS


@lru_cache(maxsize=None)
def _charger(chemin):
    return joblib.load(chemin)


def predire(retards_12m, montant_facture, anciennete_mois,
            chemin="modele_v2_avec_regles.joblib") -> dict:
    paquet = _charger(chemin)
    ligne = pd.DataFrame([{"retards_12m": retards_12m, "montant_facture": montant_facture,
                           "anciennete_mois": anciennete_mois}])

    if paquet["avec_regles"]:                         # ----- V2 -----
        ligne = appliquer_regles(ligne)
        classe = str(ligne.loc[0, "classe_regles"])
        if ligne.loc[0, "hors_plage"]:                # RP2 : pas de prédiction ML
            return {"classe_risque": classe, "decision": DECISIONS[classe],
                    "score_regles": None, "proba_impaye": None}
        proba = float(paquet["modele"].predict_proba(ligne[paquet["features"]])[0, 1])
        return {"classe_risque": classe,              # règles : RP2 > RP1 > score
                "decision": DECISIONS[classe],
                "score_regles": int(ligne.loc[0, "score_regles"]),
                "proba_impaye": round(proba, 3),      # estimation du modèle
                "impaye_predit": int(proba >= paquet["seuil"])}

    if ligne.isna().any(axis=None):                   # ----- V1 -----
        raise ValueError("Valeur manquante : prédiction impossible")
    proba = float(paquet["modele"].predict_proba(ligne[paquet["features"]])[0, 1])
    return {"proba_impaye": round(proba, 3), "impaye_predit": int(proba >= paquet["seuil"])}


if __name__ == "__main__":
    for cas in [(4, 300_000, 8), (0, 50_000, 90), (2, 150_000, 40), (15, 150_000, 40)]:
        print(cas, "->", predire(*cas))