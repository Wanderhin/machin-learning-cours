"""Partie commune aux deux versions : split, comparaison des modèles, évaluation, sauvegarde.

Adapté aux millions de lignes :
  - pas de validation croisée (trop lente) : split train / validation / test ;
  - le seuil de décision est optimisé sur la validation pour maximiser le score F1 ;
  - le KNN (qui ne passe pas à l'échelle) est entraîné et évalué sur des échantillons.
"""
from datetime import datetime

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GRAINE = 42
KNN_TRAIN_MAX = 100_000   # lignes d'entraînement du KNN
KNN_EVAL_MAX = 50_000     # lignes de validation / test du KNN


def _seuil_optimal(y_true, proba):
    """Seuil de probabilité qui maximise le F1."""
    p, r, seuils = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(seuils[i]), float(f1[i])


def _candidats(inclure_knn):
    c = {
        "Régression logistique": dict(
            modele=Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=1000))]),
            n_train=None, n_eval=None),
        "Gradient Boosting": dict(
            modele=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                  early_stopping=True, random_state=GRAINE),
            n_train=None, n_eval=None),
    }
    if inclure_knn:
        c["KNN (sur échantillon)"] = dict(
            modele=Pipeline([("scaler", StandardScaler()),
                             ("clf", KNeighborsClassifier(n_neighbors=51, n_jobs=-1))]),
            n_train=KNN_TRAIN_MAX, n_eval=KNN_EVAL_MAX)
    return c


def entrainer(X, y, features, sortie, nom_version, avec_regles,
              classe_regles=None, inclure_knn=True):
    """X : DataFrame des variables ; y : cible ; classe_regles : Series (V2) pour la référence 'règles seules'."""
    print(f"\n===== {nom_version} | {len(features)} variables : {features} =====")

    # Même graine => V1 et V2 utilisent exactement les mêmes lignes train / val / test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=GRAINE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.125, stratify=y_tmp, random_state=GRAINE)   # 70 / 10 / 20
    print(f"Train {len(X_train):,} | Validation {len(X_val):,} | Test {len(X_test):,}")

    f1_regles = None
    if classe_regles is not None:
        y_reg = (classe_regles.loc[X_test.index] == "Élevé").astype(int)
        f1_regles = f1_score(y_test, y_reg)
        print(f"Référence, règles seules ('Élevé' = impayé) : F1 test = {f1_regles:.3f}")

    # ---- Comparaison des modèles sur la validation ----
    resultats = {}
    for nom, spec in _candidats(inclure_knn).items():
        n_tr, n_ev = spec["n_train"], spec["n_eval"]
        Xt, yt = (X_train.iloc[:n_tr], y_train.iloc[:n_tr]) if n_tr else (X_train, y_train)
        Xv, yv = (X_val.iloc[:n_ev], y_val.iloc[:n_ev]) if n_ev else (X_val, y_val)
        modele = clone(spec["modele"]).fit(Xt, yt)
        seuil, f1_val = _seuil_optimal(yv, modele.predict_proba(Xv)[:, 1])
        resultats[nom] = dict(modele=modele, spec=spec, seuil=seuil, f1_val=f1_val)
        print(f"{nom:<24} F1 validation = {f1_val:.3f} (seuil {seuil:.2f})")

    meilleur_nom = max(resultats, key=lambda n: resultats[n]["f1_val"])   # choisi sur la validation
    r = resultats[meilleur_nom]
    print(f"\n>>> Meilleur modèle : {meilleur_nom}")

    # ---- Évaluation finale sur le test (jamais utilisé pour choisir) ----
    n_ev = r["spec"]["n_eval"]
    Xte, yte = (X_test.iloc[:n_ev], y_test.iloc[:n_ev]) if n_ev else (X_test, y_test)
    proba = r["modele"].predict_proba(Xte)[:, 1]
    pred = (proba >= r["seuil"]).astype(int)
    f1_test, auc = f1_score(yte, pred), roc_auc_score(yte, proba)
    print("\nMatrice de confusion (lignes = réel, colonnes = prédit) :")
    print(confusion_matrix(yte, pred))
    print(classification_report(yte, pred, target_names=["payé", "impayé"], digits=3))
    print(f"F1 test = {f1_test:.3f} | AUC = {auc:.3f}")

    # ---- Sauvegarde : ré-entraînement sur train + validation, avec le seuil retenu ----
    n_tr = r["spec"]["n_train"]
    Xf, yf = (X_tmp.iloc[:n_tr], y_tmp.iloc[:n_tr]) if n_tr else (X_tmp, y_tmp)
    modele_final = clone(r["spec"]["modele"]).fit(Xf, yf)
    joblib.dump({
        "modele": modele_final,
        "seuil": r["seuil"],
        "features": features,
        "avec_regles": avec_regles,
        "nom_modele": meilleur_nom,
        "version": nom_version,
        "metriques_test": {"f1": float(f1_test), "auc": float(auc),
                           "f1_regles_seules": None if f1_regles is None else float(f1_regles)},
        "date_entrainement": datetime.now().isoformat(timespec="seconds"),
    }, sortie)
    print(f"Modèle sauvegardé : {sortie}")
    return {"modele": meilleur_nom, "f1_test": f1_test, "auc": auc, "f1_regles": f1_regles}