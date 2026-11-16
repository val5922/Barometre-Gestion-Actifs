"""
Étape 4 et 5 : normalisation, pondération et score global.

Lit data/donnees_brutes.csv, transforme chaque indicateur en note de 0 à 100
(100 = environnement favorable aux marchés, 0 = environnement très dégradé),
puis calcule le score global comme moyenne pondérée.

Résultat : data/score.csv (les 7 notes + le score global, jour par jour).
"""

import json
import math
from pathlib import Path

import pandas as pd

DOSSIER_DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Les 4 méthodes de normalisation demandées par la consigne
# ---------------------------------------------------------------------------

def min_max(serie, borne_0, borne_100):
    """Min-Max à bornes fixes : borne_0 donne 0, borne_100 donne 100.
    Si borne_0 > borne_100, l'échelle est inversée (indicateur de risque)."""
    note = (serie - borne_0) / (borne_100 - borne_0) * 100
    return note.clip(0, 100)


def z_score(serie, inverse=False):
    """Z-score : écart à la moyenne en nombre d'écarts-types,
    converti en percentile (0-100) avec la loi normale."""
    z = (serie - serie.mean()) / serie.std()
    if inverse:
        z = -z
    return z.apply(lambda x: 50 * (1 + math.erf(x / math.sqrt(2))) if pd.notna(x) else x)


def cible(serie, objectif, points_par_unite):
    """Distance à une cible : 100 pile sur l'objectif, on perd des points
    à mesure qu'on s'en éloigne (dans un sens comme dans l'autre)."""
    return (100 - (serie - objectif).abs() * points_par_unite).clip(0, 100)


# ---------------------------------------------------------------------------
# Paramètres du baromètre
# ---------------------------------------------------------------------------

POIDS = {
    "VIX": 20,
    "SP500": 15,
    "CAC40": 10,
    "COURBE_10A_2A": 15,
    "SPREAD_HY": 15,
    "INFLATION": 10,
    "TAUX_2A": 15,
}


def calculer_notes(brut):
    notes = pd.DataFrame(index=brut.index)

    # 1. VIX : Min-Max inversé. 12 = marché serein (100), 35 = panique (0).
    notes["VIX"] = min_max(brut["VIX"], borne_0=35, borne_100=12)

    # 2-3. Actions : écart du cours à sa moyenne mobile 200 jours (tendance),
    #      normalisé par Z-score.
    for indice in ["SP500", "CAC40"]:
        mm200 = brut[indice].rolling(200).mean()
        ecart = (brut[indice] / mm200 - 1) * 100
        notes[indice] = z_score(ecart)

    # 4. Courbe 10 ans - 2 ans : Min-Max. Inversée à -1 pt = 0, +1,5 pt = 100.
    notes["COURBE_10A_2A"] = min_max(brut["COURBE_10A_2A"], borne_0=-1, borne_100=1.5)

    # 5. Spread High Yield : Min-Max inversé. 7 % = stress (0), 2,5 % = calme (100).
    notes["SPREAD_HY"] = min_max(brut["SPREAD_HY"], borne_0=7, borne_100=2.5)

    # 6. Inflation : CPI transformé en glissement annuel (%), puis méthode cible.
    #    2 % = 100, on perd 25 points par point d'écart (4 % -> 50, 6 % -> 0).
    cpi_mensuel = brut["CPI"].resample("MS").last()
    inflation = (cpi_mensuel.pct_change(12) * 100).reindex(brut.index, method="ffill")
    notes["INFLATION"] = cible(inflation, objectif=2, points_par_unite=25)

    # 7. Taux 2 ans : variation sur 6 mois (~126 jours de bourse) = ce que le
    #    marché anticipe de la Fed. Hausse = resserrement attendu = négatif.
    #    Z-score inversé.
    variation = brut["TAUX_2A"] - brut["TAUX_2A"].shift(126)
    notes["TAUX_2A"] = z_score(variation, inverse=True)

    return notes, inflation


def score_global(notes):
    """Moyenne pondérée. Si un indicateur manque un jour donné (ex. spread HY
    avant 2023), les poids des indicateurs disponibles sont rééquilibrés."""
    poids = pd.Series(POIDS)
    disponibles = notes[poids.index].notna()
    somme_poids = (disponibles * poids).sum(axis=1)
    somme_notes = (notes[poids.index].fillna(0) * poids).sum(axis=1)
    score = somme_notes / somme_poids
    # On n'affiche le score que si au moins 80 % des poids sont disponibles.
    return score.where(somme_poids >= 80)


def exporter_site(brut, notes, inflation):
    """Écrit public/data.json, le fichier lu par le site web."""
    debut = notes.index[0]
    dernier = notes.index[-1]
    valeurs_brutes = {
        "VIX": round(brut.loc[dernier, "VIX"], 1),
        "SP500": round(brut.loc[dernier, "SP500"], 0),
        "CAC40": round(brut.loc[dernier, "CAC40"], 0),
        "COURBE_10A_2A": round(brut.loc[dernier, "COURBE_10A_2A"], 2),
        "SPREAD_HY": round(brut.loc[dernier, "SPREAD_HY"], 2),
        "INFLATION": round(inflation.loc[dernier], 2),
        "TAUX_2A": round(brut.loc[dernier, "TAUX_2A"], 2),
    }
    data = {
        "mise_a_jour": dernier.strftime("%Y-%m-%d"),
        "debut": debut.strftime("%Y-%m-%d"),
        "poids": POIDS,
        "valeurs_brutes": valeurs_brutes,
        "dates": [d.strftime("%Y-%m-%d") for d in notes.index],
        "score": notes["SCORE"].round(1).tolist(),
        # Les valeurs manquantes deviennent null (le format JSON refuse NaN).
        "notes": {nom: [None if pd.isna(x) else x for x in notes[nom].round(1)]
                  for nom in POIDS},
    }
    dossier = Path(__file__).parent / "public"
    dossier.mkdir(exist_ok=True)
    with open(dossier / "data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def main():
    brut = pd.read_csv(DOSSIER_DATA / "donnees_brutes.csv", index_col=0, parse_dates=True)
    notes, inflation = calculer_notes(brut)
    notes["SCORE"] = score_global(notes)
    notes = notes.dropna(subset=["SCORE"])

    notes.round(1).to_csv(DOSSIER_DATA / "score.csv")
    exporter_site(brut, notes, inflation)

    dernier = notes.iloc[-1]
    print(f"Score du {notes.index[-1].date()} : {dernier['SCORE']:.0f} / 100\n")
    print(f"{'Indicateur':<15}{'Note':>6}{'Poids':>7}{'Contribution':>14}")
    for nom, p in POIDS.items():
        print(f"{nom:<15}{dernier[nom]:>6.0f}{p:>6}%{dernier[nom] * p / 100:>14.1f}")
    print(f"\nInflation US actuelle : {inflation.iloc[-1]:.2f} %")
    print(f"Historique : {notes.index[0].date()} -> {notes.index[-1].date()}")
    print(f"Score min {notes['SCORE'].min():.0f} ({notes['SCORE'].idxmin().date()}), "
          f"max {notes['SCORE'].max():.0f} ({notes['SCORE'].idxmax().date()})")


if __name__ == "__main__":
    main()
