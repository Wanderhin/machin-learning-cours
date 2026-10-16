# Évaluation du risque d'impayé avec UNIQUEMENT les règles métiers (pas de dataset, pas de machine learning)

# %%
# ---------------------------------------------------------------
# 1) PARAMÈTRES : les seuils des règles (propositions à valider avec le métier)
# ---------------------------------------------------------------
MONTANT_MOYEN = 100_000   # FCFA : en dessous = 0 point
MONTANT_ELEVE = 250_000   # FCFA : au-dessus = 2 points
ANCIENNETE_1_AN = 12      # mois : moins d'1 an = "nouveau client"
ANCIENNETE_2_ANS = 24     # mois
ANCIENNETE_5_ANS = 60     # mois

# Décision associée à chaque classe de risque
DECISIONS = {
    "Faible": "Facturation normale",
    "Moyen": "Rappel avant échéance, suivi renforcé",
    "Élevé": "Paiement anticipé ou garantie, validation par un gestionnaire",
    "Traitement manuel": "Traitement manuel (données manquantes ou hors plage)",
}


# %%
# ---------------------------------------------------------------
# 2) GRILLE DE POINTS : une fonction par variable
# ---------------------------------------------------------------
def points_retards(retards):
    """Retards de paiement sur 12 mois (facteur le plus important)."""
    if retards == 0:
        return 0
    if retards <= 2:      # 1 à 2 retards
        return 1
    if retards == 3:
        return 3
    return 5              # 4 retards et plus


def points_montant(montant):
    """Montant de la facture en FCFA."""
    if montant < MONTANT_MOYEN:       # moins de 100 000
        return 0
    if montant <= MONTANT_ELEVE:      # de 100 000 à 250 000
        return 1
    return 2                          # plus de 250 000


def points_anciennete(anciennete_mois):
    """Ancienneté du client (en mois)."""
    if anciennete_mois > ANCIENNETE_5_ANS:      # plus de 5 ans
        return 0
    if anciennete_mois >= ANCIENNETE_2_ANS:     # de 2 à 5 ans
        return 1
    return 2                                    # moins de 2 ans


# %%
# ---------------------------------------------------------------
# 3) RÈGLES DE PRIORITÉ (elles passent avant le score)
# ---------------------------------------------------------------
def est_hors_plage(retards, montant, anciennete_mois):
    """RP2 : valeur manquante ou hors plage -> traitement manuel."""
    if retards is None or montant is None or anciennete_mois is None:
        return True
    return retards < 0 or retards > 12 or montant <= 0 or anciennete_mois < 0


def est_rp1(montant, anciennete_mois):
    """RP1 : client de moins d'1 an avec une facture > 250 000 FCFA -> risque élevé."""
    return anciennete_mois < ANCIENNETE_1_AN and montant > MONTANT_ELEVE


# %%
# ---------------------------------------------------------------
# 4) ÉVALUATION D'UN CLIENT : points -> score -> classe -> décision
# ---------------------------------------------------------------
def evaluer_client(retards, montant, anciennete_mois):
    # RP2 d'abord : si les données sont invalides, on ne calcule aucun score
    if est_hors_plage(retards, montant, anciennete_mois):
        classe = "Traitement manuel"
        return {"score": None, "classe_risque": classe, "decision": DECISIONS[classe],
                "regle_appliquee": "RP2", "detail_points": None}

    # Calcul du score = somme des points (de 0 à 9)
    p_ret = points_retards(retards)
    p_mon = points_montant(montant)
    p_anc = points_anciennete(anciennete_mois)
    score = p_ret + p_mon + p_anc

    # Classe de risque : RP1 passe avant le score
    if est_rp1(montant, anciennete_mois):
        classe, regle = "Élevé", "RP1"
    elif score >= 5:
        classe, regle = "Élevé", "Score"
    elif score >= 3:
        classe, regle = "Moyen", "Score"
    else:
        classe, regle = "Faible", "Score"

    return {"score": score, "classe_risque": classe, "decision": DECISIONS[classe],
            "regle_appliquee": regle,
            "detail_points": f"retards={p_ret} + montant={p_mon} + ancienneté={p_anc}"}


# %%
# ---------------------------------------------------------------
# 5) AFFICHAGE DU SCORE pour quelques clients exemples
# (à remplacer par tes propres clients : retards sur 12 mois, montant en FCFA, ancienneté en mois)
# ---------------------------------------------------------------
clients = [
    {"nom": "Client A", "retards": 0, "montant": 50_000,  "anciennete": 90},   # fiable
    {"nom": "Client B", "retards": 2, "montant": 150_000, "anciennete": 40},   # intermédiaire
    {"nom": "Client C", "retards": 4, "montant": 300_000, "anciennete": 8},    # nouveau + grosse facture (RP1)
    {"nom": "Client D", "retards": 3, "montant": 120_000, "anciennete": 30},   # plusieurs retards
    {"nom": "Client E", "retards": 15, "montant": 150_000, "anciennete": 40},  # retards hors plage (RP2)
]

for c in clients:
    r = evaluer_client(c["retards"], c["montant"], c["anciennete"])
    print(f"--- {c['nom']} ---")
    print(f"Retards : {c['retards']} | Montant : {c['montant']:,} FCFA | Ancienneté : {c['anciennete']} mois")
    if r["score"] is None:
        print("Score : non calculé (données invalides)")
    else:
        print(f"Score : {r['score']} / 9  ({r['detail_points']})")
    print(f"Classe de risque : {r['classe_risque']}  (règle : {r['regle_appliquee']})")
    print(f"Décision : {r['decision']}\n")


# %%
# ---------------------------------------------------------------
# 6) (OPTIONNEL) SAISIE D'UN CLIENT AU CLAVIER
# ---------------------------------------------------------------
if input("Tester un client au clavier ? (o/n) : ").strip().lower() == "o":
    retards = int(input("Retards de paiement sur 12 mois : "))
    montant = float(input("Montant de la facture (FCFA) : "))
    anciennete = int(input("Ancienneté (mois) : "))
    r = evaluer_client(retards, montant, anciennete)
    print("\nScore :", r["score"], "/ 9" if r["score"] is not None else "")
    print("Classe de risque :", r["classe_risque"], f"(règle : {r['regle_appliquee']})")
    print("Décision :", r["decision"])