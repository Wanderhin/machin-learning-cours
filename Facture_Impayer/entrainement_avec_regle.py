"""VERSION 2 : données + règles métiers (points, score, RP1, RP2)."""
from Facture_Impayer.Chargement import charger_donnees
from Facture_Impayer.entrainement_commun import entrainer
from Facture_Impayer.regle_metier import appliquer_regles, FEATURES_ML

NB_LIGNES_MAX = None      # None = les 5 millions ; ex. 300_000 pour tester vite

df = charger_donnees(NB_LIGNES_MAX)
df = appliquer_regles(df)

# RP2 : valeurs manquantes ou hors plage -> traitement manuel, exclues de l'apprentissage
nb_manuel = int(df["hors_plage"].sum())
print(f"RP2 (traitement manuel, exclues) : {nb_manuel:,}")
df = df[~df["hors_plage"]]

print("\nClasses de risque (règles) et taux d'impayés réel :")
print(df.groupby("classe_regles", observed=True)["impaye"]
        .agg(effectif="count", taux_impaye="mean").round(3))

entrainer(
    X=df[FEATURES_ML],
    y=df["impaye"],
    features=FEATURES_ML,
    sortie="modele_v2_avec_regles.joblib",
    nom_version="V2 - avec règles métiers",
    avec_regles=True,
    classe_regles=df["classe_regles"],
)