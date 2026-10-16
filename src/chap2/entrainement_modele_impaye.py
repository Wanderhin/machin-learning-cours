"""
Entraînement, évaluation et sauvegarde du modèle de prédiction des impayés.

Utilisation (depuis PyCharm ou le terminal) :
    python entrainement_impayes.py
    python entrainement_impayes.py chemin/vers/fichier.xlsx

Structure attendue du projet :
    projet/
      data/donnees_impayes_Seance_01.xlsx
      src/entrainement_impayes.py      <- ce fichier
    Les dossiers models/ et reports/ sont créés automatiquement.
"""
import json
import sys
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")  # sauvegarde les figures en PNG, sans fenêtre (évite les bugs de backend)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------
from pathlib import Path

DOSSIER_SCRIPT = Path(__file__).resolve().parent     # .../src/chap2
RACINE = DOSSIER_SCRIPT.parents[1]                   # racine du projet (marchin_learning_projet_dev)

FICHIER = Path(sys.argv[1]) if len(sys.argv) > 1 else DOSSIER_SCRIPT / "data" / "donnees_impayes_Seance_01.xlsx"

DOSSIER_MODELES = RACINE / "models"
DOSSIER_RAPPORTS = RACINE / "reports"
DOSSIER_MODELES.mkdir(exist_ok=True)      # ces deux lignes créent les dossiers
DOSSIER_RAPPORTS.mkdir(exist_ok=True)

CIBLE = "impaye"
SEED = 42

# ----------------------------------------------------------------------------
# 2. Chargement et contrôle des données
# ----------------------------------------------------------------------------
df = pd.read_excel(FICHIER, sheet_name="Donnees")
print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("Valeurs manquantes :", int(df.isna().sum().sum()))
print("Doublons           :", int(df.duplicated().sum()))
print("Répartition de la cible :\n", df[CIBLE].value_counts(normalize=True).round(3))

X = df.drop(columns=CIBLE)
y = df[CIBLE]
variables = list(X.columns)

# Corrélation avec la cible (pour le rapport)
correlations = df.corr()[CIBLE].drop(CIBLE).sort_values()
print("\nCorrélation avec la cible :\n", correlations.round(3))

# ----------------------------------------------------------------------------
# 3. Séparation 80 % / 20 % (stratifiée)
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\nTrain : {len(X_train)} lignes | Test : {len(X_test)} lignes")

# ----------------------------------------------------------------------------
# 4. Comparaison de modèles (recherche d'hyperparamètres + validation croisée)
#    Tout se fait sur le train_set uniquement.
# ----------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

candidats = {
    "Régression logistique": (
        Pipeline([("scaler", StandardScaler()),
                  ("model", LogisticRegression(max_iter=1000, random_state=SEED))]),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "SVM": (
        Pipeline([("scaler", StandardScaler()),
                  ("model", SVC(probability=True, random_state=SEED))]),
        {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
    ),
    "Random Forest": (
        Pipeline([("model", RandomForestClassifier(n_estimators=300, random_state=SEED))]),
        {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [1, 5, 10]},
    ),
    "XGBoost": (
        Pipeline([("model", XGBClassifier(n_estimators=200, subsample=0.8,
                                          colsample_bytree=0.8, eval_metric="logloss",
                                          random_state=SEED))]),
        {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.1]},
    ),
}

resultats = {}
meilleurs_estimateurs = {}
for nom, (pipe, grille) in candidats.items():
    gs = GridSearchCV(pipe, grille, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    resultats[nom] = {"auc_cv": round(gs.best_score_, 4),
                      "meilleurs_parametres": {k: (v if isinstance(v, (int, float, str)) else str(v))
                                               for k, v in gs.best_params_.items()}}
    meilleurs_estimateurs[nom] = gs.best_estimator_
    print(f"{nom:24s} AUC CV = {gs.best_score_:.4f}  {gs.best_params_}")

# Choix du meilleur modèle : AUC la plus élevée en validation croisée.
# En cas d'égalité (à 0,005 près), on préfère le modèle le plus simple (ordre du dictionnaire).
meilleur_auc = max(r["auc_cv"] for r in resultats.values())
nom_meilleur = next(n for n, r in resultats.items() if r["auc_cv"] >= meilleur_auc - 0.005)
modele = meilleurs_estimateurs[nom_meilleur]
print(f"\n>>> Modèle retenu : {nom_meilleur}")

# ----------------------------------------------------------------------------
# 5. Évaluation finale sur le test_set (utilisé une seule fois)
# ----------------------------------------------------------------------------
y_pred = modele.predict(X_test)
y_proba = modele.predict_proba(X_test)[:, 1]

metriques_test = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1": round(f1_score(y_test, y_pred), 4),
    "auc": round(roc_auc_score(y_test, y_proba), 4),
}
print("\nMétriques sur le test_set :", metriques_test)
print("\nMatrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=["payé", "impayé"]))

# ----------------------------------------------------------------------------
# 6. Graphiques
# ----------------------------------------------------------------------------
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["payé", "impayé"])
plt.title(f"Matrice de confusion - {nom_meilleur}")
plt.tight_layout()
plt.savefig(DOSSIER_RAPPORTS / "matrice_confusion.png", dpi=150)
plt.close()

RocCurveDisplay.from_predictions(y_test, y_proba, name=nom_meilleur)
plt.plot([0, 1], [0, 1], "--", color="grey")
plt.title("Courbe ROC (test_set)")
plt.tight_layout()
plt.savefig(DOSSIER_RAPPORTS / "courbe_roc.png", dpi=150)
plt.close()

# Importance des variables
etape = modele.named_steps["model"]
if hasattr(etape, "coef_") and etape.coef_.shape[0] == 1:
    importance = pd.Series(etape.coef_[0], index=variables).sort_values()
    titre = "Coefficients (variables normalisées)"
elif hasattr(etape, "feature_importances_"):
    importance = pd.Series(etape.feature_importances_, index=variables).sort_values()
    titre = "Importance des variables"
else:
    importance = None

if importance is not None:
    importance.plot(kind="barh")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(titre)
    plt.tight_layout()
    plt.savefig(DOSSIER_RAPPORTS / "importance_variables.png", dpi=150)
    plt.close()
    print("\n", titre, ":\n", importance.round(3))

# ----------------------------------------------------------------------------
# 7. Sauvegarde du modèle et de toutes les informations utiles au déploiement
# ----------------------------------------------------------------------------
# Le Pipeline contient déjà la normalisation : il suffit de le recharger et
# d'appeler predict / predict_proba sur un DataFrame avec les mêmes colonnes.
chemin_modele = DOSSIER_MODELES / "modele_impaye.joblib"
joblib.dump(modele, chemin_modele)

metadonnees = {
    "date_entrainement": datetime.now().isoformat(timespec="seconds"),
    "modele": nom_meilleur,
    "parametres": resultats[nom_meilleur]["meilleurs_parametres"],
    "cible": CIBLE,
    "signification_cible": {"0": "facture payée", "1": "facture impayée"},
    "variables_entree": variables,
    "types_variables": {c: str(t) for c, t in X.dtypes.items()},
    "plages_valeurs_entrainement": {
        c: {"min": float(X[c].min()), "max": float(X[c].max()), "moyenne": round(float(X[c].mean()), 3)}
        for c in variables
    },
    "seuil_decision": 0.5,
    "taille_train": int(len(X_train)),
    "taille_test": int(len(X_test)),
    "comparaison_modeles_cv": resultats,
    "metriques_test": metriques_test,
    "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                 "pandas": pd.__version__, "numpy": np.__version__},
    "fichier_modele": chemin_modele.name,
}
with open(DOSSIER_MODELES / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadonnees, f, ensure_ascii=False, indent=2)

print(f"\nModèle sauvegardé  : {chemin_modele}")
print(f"Métadonnées        : {DOSSIER_MODELES / 'metadata.json'}")
print(f"Graphiques         : {DOSSIER_RAPPORTS}")

# ----------------------------------------------------------------------------
# 8. Test de rechargement et exemple de prédiction (comme en production)
# ----------------------------------------------------------------------------
modele_charge = joblib.load(chemin_modele)
nouveau_client = pd.DataFrame([{
    "consommation_kwh": 180.0,
    "anciennete_mois": 24,
    "retards_12m": 3,
    "montant_facture": 30000.0,
}])[variables]  # même ordre de colonnes qu'à l'entraînement
proba = modele_charge.predict_proba(nouveau_client)[0, 1]
print(f"\nExemple : probabilité d'impayé = {proba:.1%} -> "
      f"{'IMPAYÉ' if proba >= metadonnees['seuil_decision'] else 'PAYÉ'}")