"""VERSION 1 : uniquement les données (les 3 variables brutes, aucune règle métier)."""
from Facture_Impayer.Chargement import charger_donnees
from entrainement_commun import entrainer

NB_LIGNES_MAX = None      # None = les 5 millions ; ex. 300_000 pour tester vite
COLONNES_BRUTES = ["retards_12m", "montant_facture", "anciennete_mois"]

df = charger_donnees(NB_LIGNES_MAX)
df = df.dropna(subset=COLONNES_BRUTES)          # le ML ne gère pas les valeurs manquantes ici

entrainer(
    X=df[COLONNES_BRUTES],
    y=df["impaye"],
    features=COLONNES_BRUTES,
    sortie="modele_v1_sans_regles.joblib",
    nom_version="V1 - sans règles métiers",
    avec_regles=False,
)