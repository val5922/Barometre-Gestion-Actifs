"""
Étape 3 — Collecte automatique des données du baromètre.

Télécharge 7 séries (5 ans d'historique) depuis deux sources ouvertes :
  - Yahoo Finance (plateforme financière) : VIX, S&P 500, CAC 40
  - FRED, Réserve fédérale de St. Louis (base publique) : courbe des taux,
    spread High Yield, inflation CPI, taux 2 ans US

Résultat : data/donnees_brutes.csv (une ligne par jour, une colonne par indicateur).
Aucune clé d'API nécessaire.
"""

import io
import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DEBUT = (date.today() - timedelta(days=5 * 365)).isoformat()
DOSSIER_DATA = Path(__file__).parent / "data"

YAHOO = {
    "VIX": "^VIX",
    "SP500": "^GSPC",
    "CAC40": "^FCHI",
}

FRED = {
    "COURBE_10A_2A": "T10Y2Y",
    "SPREAD_HY": "BAMLH0A0HYM2",
    "CPI": "CPIAUCSL",
    "TAUX_2A": "DGS2",
}


def telecharger(url):
    # Yahoo exige un identifiant "navigateur", FRED le bloque : on adapte.
    agent = "Mozilla/5.0" if "yahoo" in url else "curl/8.7.1"
    requete = urllib.request.Request(url, headers={"User-Agent": agent})
    with urllib.request.urlopen(requete, timeout=30) as reponse:
        return reponse.read().decode("utf-8")


def serie_yahoo(symbole):
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + urllib.request.quote(symbole)
        + "?range=5y&interval=1d"
    )
    resultat = json.loads(telecharger(url))["chart"]["result"][0]
    dates = pd.to_datetime(resultat["timestamp"], unit="s").normalize()
    cloture = resultat["indicators"]["quote"][0]["close"]
    return pd.Series(cloture, index=dates).dropna()


def serie_fred(code):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={DEBUT}"
    df = pd.read_csv(io.StringIO(telecharger(url)), na_values=".")
    df.columns = ["date", "valeur"]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["valeur"].dropna()


def main():
    series = {}
    for nom, symbole in YAHOO.items():
        series[nom] = serie_yahoo(symbole)
        print(f"Yahoo  {nom:<14} {len(series[nom]):>5} valeurs")
    for nom, code in FRED.items():
        series[nom] = serie_fred(code)
        print(f"FRED   {nom:<14} {len(series[nom]):>5} valeurs")

    # On aligne tout sur un calendrier quotidien commun.
    # Le CPI est mensuel : on garde la dernière valeur connue (ffill).
    df = pd.DataFrame(series)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.ffill()
    df = df[df.index >= DEBUT]

    DOSSIER_DATA.mkdir(exist_ok=True)
    chemin = DOSSIER_DATA / "donnees_brutes.csv"
    df.to_csv(chemin, float_format="%.4f")
    print(f"\nFichier enregistré : {chemin}")
    print("\nDernières valeurs :")
    print(df.tail(3).round(2).to_string())


if __name__ == "__main__":
    main()
