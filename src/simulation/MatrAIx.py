import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.Stochastisch import REWARDS, _cancel

DATEN = Path(__file__).resolve().parents[2] / "matrAIx" / "daten"
PUFFER = DATEN / "reaktionen.jsonl"

PREISE = ["niedrig", "mittel", "hoch"]
SALE = ["nein", "ja"]
ZUSTAENDE = ["kein Grund", "Grund"]
P_GRUND_RETOURE, P_GRUND_BEHALTEN = 0.5, 0.2
WARNUNGEN = ["warnung", "hinweis"]   # nur "warnung" wird verwendet
STUFEN_P = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}


def kontext_code(persona, preis, sale):
    return (persona * len(PREISE) + preis) * len(SALE) + sale


def kontext_teile(code):
    code, sale = divmod(code, len(SALE))
    persona, preis = divmod(code, len(PREISE))
    return persona, preis, sale


def lade_kontext(split, variante="A"):
    return np.load(DATEN / f"kontext_{split}.npz")[variante]


def lade_puffer(pfad=None):
    pfad = Path(pfad or PUFFER)
    if not pfad.exists():
        return pd.DataFrame()
    return pd.read_json(pfad, lines=True, dtype=False).drop_duplicates(["custom_id", "modell", "prompt_version"])


def gueltige_antworten(df, aufgabe="haupt", modell=None, prompt_version=None):
    if df.empty:
        return df
    df = df[(df.aufgabe == aufgabe) & (df.status == "ok")]
    for spalte, wahl in [("modell", modell), ("prompt_version", prompt_version)]:
        if wahl is not None:
            df = df[df[spalte] == wahl]
        elif df[spalte].nunique() > 1:
            raise ValueError(f"Puffer enthält mehrere Werte für {spalte}: {sorted(df[spalte].unique())}, bitte wählen "
                             "(Modell: MATRAIX_MODELL in Experimente.py) oder alte Zeilen aus dem Puffer entfernen")
    return df


def abbruch_tabelle(df, n_personas, stufen_p=None, warnung="warnung"):
    stufen_p = stufen_p or STUFEN_P
    tabelle = np.full((n_personas, len(PREISE), len(SALE), len(ZUSTAENDE)), np.nan)
    if df.empty:
        return tabelle
    df = df[df.warnung == warnung]
    p = df.response.map(stufen_p)
    mittel = p.groupby([df.persona.astype(int), df.preis, df.sale, df.zustand]).mean()
    for (persona, preis, sale, zustand), wert in mittel.items():
        tabelle[persona, preis, sale, zustand] = wert
    return tabelle


def vergroebern(tabelle):
    voll = tabelle.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for achsen in [(2,), (1, 2), (0, 1, 2)]:
            ersatz = np.nanmean(tabelle, axis=achsen, keepdims=True)
            voll = np.where(np.isnan(voll), np.broadcast_to(ersatz, voll.shape), voll)
    return voll


class MatrAIx:
    def __init__(self, rewards="basis", puffer=None, modell=None, prompt_version=None, stufen_p=None,
                 warnung="warnung", p_grund_retoure=P_GRUND_RETOURE, p_grund_behalten=P_GRUND_BEHALTEN,
                 min_abdeckung=0.99):
        n_personas = len(json.loads((DATEN / "kohorte.json").read_text())["personas"])
        antworten = gueltige_antworten(lade_puffer(puffer), "haupt", modell, prompt_version)
        if antworten.empty:
            raise RuntimeError(f"Puffer {puffer or PUFFER} enthält keine gültigen Antworten (Modell {modell}), "
                               "erst Puffer.py fuellen")
        tabelle = abbruch_tabelle(antworten, n_personas, stufen_p, warnung)
        self.abdeckung = float(np.mean(~np.isnan(tabelle)))
        if self.abdeckung < min_abdeckung:
            raise RuntimeError(f"Puffer deckt erst {self.abdeckung:.1%} der Schlüssel ab (mindestens "
                               f"{min_abdeckung:.0%}), erst Puffer.py fuellen")
        self.p = vergroebern(tabelle)
        self.p_zustand = self.p.mean(axis=(0, 1, 2))
        self.p_grund_retoure = p_grund_retoure
        self.p_grund_behalten = p_grund_behalten
        self.rewards = REWARDS[rewards]

    def evaluate(self, truth, action, context):
        returned = truth == 1
        if action == 0:
            outcome = 'NoWarn_Returned' if returned else 'NoWarn_NotReturned'
        elif action == 1:
            p = self.p[kontext_teile(int(context))] if context >= 0 else self.p_zustand
            g = self.p_grund_retoure if returned else self.p_grund_behalten
            outcome = ('Warn_Returned' if returned else 'Warn_NotReturned') + _cancel(g * p[1] + (1 - g) * p[0])
        else:
            raise ValueError(f"Ungültige Aktion {action}, erwartet 0 oder 1")
        return self.rewards[outcome], outcome
