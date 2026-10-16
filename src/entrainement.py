"""
Prédiction du retard de projets logiciels avec XGBoost
Jeu de données : jeu_donnees_retards_projets_ML_1000.xlsx (feuille "Donnees")
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             classification_report, confusion_matrix)
from scipy.stats import randint, uniform
from xgboost import XGBClassifier, plot_importance

SEED = 42
base_dir = Path(__file__).parent.absolute()
FICHIER = base_dir.parent / "data" / "jeu_donnees_retards_projets_ML_1000.xlsx"


# ------------------------------------------------------------------
# 1. Chargement
# ------------------------------------------------------------------
df = pd.read_excel(FICHIER, sheet_name="Donnees")
print(df.shape)
print(df["retard"].value_counts(normalize=True))   # ~35 % de retards

# ------------------------------------------------------------------
# 2. Choix des variables
#    - id_projet      : simple identifiant, aucune information
#    - retard_libelle : copie exacte de la cible  -> fuite de données
#    - retard_jours   : connu seulement APRÈS la fin du projet
#                       (0 si à temps) -> fuite de données
# ------------------------------------------------------------------
X = df.drop(columns=["id_projet", "retard", "retard_libelle", "retard_jours"])
y = df["retard"]

# ------------------------------------------------------------------
# 3. Séparation train / validation / test (stratifiée)
# ------------------------------------------------------------------
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.20, stratify=y_tmp, random_state=SEED)

# Classes un peu déséquilibrées : nb négatifs / nb positifs
spw = (y_train == 0).sum() / (y_train == 1).sum()

# ------------------------------------------------------------------
# 4. Modèle de référence (hyperparamètres par défaut)
# ------------------------------------------------------------------
base = XGBClassifier(random_state=SEED, eval_metric="logloss")
base.fit(X_train, y_train)
print("AUC base (val) :", roc_auc_score(y_val, base.predict_proba(X_val)[:, 1]))

# ------------------------------------------------------------------
# 5. Recherche des hyperparamètres (validation croisée sur le train)
#    Petit jeu de données -> arbres peu profonds, learning_rate faible
# ------------------------------------------------------------------
param_dist = {
    "n_estimators":     randint(100, 500),
    "learning_rate":    uniform(0.01, 0.14),     # 0.01 à 0.15
    "max_depth":        randint(2, 6),
    "min_child_weight": randint(1, 10),
    "subsample":        uniform(0.6, 0.4),       # 0.6 à 1.0
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma":            uniform(0, 3),
    "reg_lambda":       uniform(1, 9),           # 1 à 10
    "reg_alpha":        uniform(0, 2),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(
    XGBClassifier(random_state=SEED, eval_metric="logloss",
                  scale_pos_weight=spw),
    param_distributions=param_dist,
    n_iter=60, cv=cv, scoring="roc_auc",
    n_jobs=-1, random_state=SEED, verbose=1,
)
search.fit(X_train, y_train)
print("Meilleurs hyperparamètres :", search.best_params_)
print("AUC CV :", round(search.best_score_, 4))

# ------------------------------------------------------------------
# 6. Modèle final avec early stopping sur la validation
# ------------------------------------------------------------------
best = search.best_params_.copy()
best["n_estimators"] = 1000          # l'early stopping choisira le bon nombre

final = XGBClassifier(**best, random_state=SEED, eval_metric="logloss",
                      scale_pos_weight=spw, early_stopping_rounds=30)
final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("Nombre d'arbres retenu :", final.best_iteration + 1)

# ------------------------------------------------------------------
# 7. Évaluation finale sur le jeu de TEST (touché une seule fois)
# ------------------------------------------------------------------
proba = final.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print("\nAccuracy :", round(accuracy_score(y_test, pred), 3))
print("AUC      :", round(roc_auc_score(y_test, proba), 3))
print("F1       :", round(f1_score(y_test, pred), 3))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["À temps", "En retard"]))

# ------------------------------------------------------------------
# 8. Importance des variables
# ------------------------------------------------------------------
plot_importance(final, importance_type="gain", height=0.6)
plt.tight_layout()
plt.savefig("importance_variables.png", dpi=150)