import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent / "src"))
from simulation.MatrAIx import (DATEN, P_GRUND_BEHALTEN, P_GRUND_RETOURE, PREISE, PUFFER, SALE, STUFEN_P, WARNUNGEN,
                                ZUSTAENDE, kontext_code, lade_puffer)

ASOS = HIER.parent / "data"
BEZUGSJAHR = 2021
ALTERSGRENZEN = [25, 35, 45, 55]
ALTERSSTUFEN = ["18-24", "25-34", "35-44", "45-54", "55+"]

K = 5
K_OHNE_PERSONA = 20
MAX_VERSUCHE = 3
MODELL = "qwen/qwen3.8-27b@deepinfra/bf16"
AKTIVE_WARNUNGEN = ["warnung"]
SEED = 2026

OPENROUTER = "https://openrouter.ai/api/v1"
MAX_TOKENS = 2000
MODELL_PARAMETER = {}   # Slug -> Felder, die die Anfrage überschreiben
STUFEN_DENKEN = ["minimal", "low", "medium", "high", "xhigh", "max"]
BATCHES = DATEN / "batches.jsonl"
BATCH_GROESSE = 10_000
WARTEN = 30
PARALLEL = 8
FEHLER_IN_FOLGE = 50

# Persona-Treue: Attribut -> (Arm hoch, Arm niedrig)
TREUE = {
    "lstyle_shopping_style": ("Researcher", "Impulse buyer"),
    "trust_level": ("Trusting", "Skeptical"),
    "economic_motivation": ("Cost-sensitive", "Premium-seeking"),
    "val_sustainability": ("Core value", "Irrelevant"),
    "time_pressure": ("Deadline", "No rush"),
}
TREUE_BASIS = 10
TREUE_SITUATIONEN = [(1, sale, zustand) for sale in (0, 1) for zustand in (0, 1)]

# Jede Änderung an den Prompt-Texten ergibt eine neue PROMPT_VERSION (die alten Antworten gelten dann nicht mehr)
SYSTEM = ("You are role-playing the person described below while they shop online. Answer as this person would, "
          "based on their background, personality, values and shopping habits. Stay in character and do not "
          "mention that you are an AI.")
SYSTEM_OHNE_PERSONA = ("You are a customer shopping online. Answer as a typical customer would. "
                       "Do not mention that you are an AI.")
SALE_TEXT = ["It has little or no discount.", "It is heavily discounted."]
ZUSTAND_TEXT = [
    "Based on your own judgement, you see no reason why you would send it back.",
    "Based on your own judgement, you see a reason why you might send it back.",
]
WARNUNG_TEXT = {
    "warnung": "Before you order: for orders like yours, this item is more likely than usual to be returned. Please "
               "check the size guide and product details to make sure it is right for you. Every return means extra "
               "transport and packaging. The choice is yours, and your return options stay the same.",
    "hinweis": "Tip: check the size guide and product details to make sure this item is right for you.",
}
STUFEN_TEXT = {1: "I complete the order as planned", 2: "I probably still order it",
               3: "I'm unsure; it could go either way", 4: "I probably remove this item from my order",
               5: "I definitely remove this item from my order"}
USER = ("You are about to check out at a large online fashion retailer. The item in your basket costs {preis}. "
        "{sale}\nAt checkout the shop shows you this message:\n\"{warnung}\"\nYou read the message and take another "
        "look at the item, the way you usually would. {zustand}\n\n"
        "How do you respond, given the message and your own judgement? Choose one level from 1 to 5:\n{stufen}")
SCHEMA = {"type": "object", "properties": {"response": {"type": "integer", "enum": sorted(STUFEN_P)}},
          "required": ["response"], "additionalProperties": False}


@lru_cache
def kohorte():
    return json.loads((DATEN / "kohorte.json").read_text())


def mischen(x):
    z = x.astype(np.int64).view(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def zuordnung():
    personas = kohorte()["personas"]
    zellen = {}
    for p in personas:
        zellen.setdefault((p["zelle"]["alter"], p["zelle"]["geschlecht"]), []).append(p["index"])
    teile = []
    for split in ("training", "testing"):
        c = pd.read_pickle(ASOS / f"customer_nodes_{split}.p")[["hash(customerId)", "yearOfBirth", "isMale", "premier"]]
        assert c["hash(customerId)"].is_unique
        alter = np.array(ALTERSSTUFEN, dtype=object)[np.digitize(BEZUGSJAHR - c.yearOfBirth.to_numpy(), ALTERSGRENZEN)]
        geschlecht = np.where(c.isMale.to_numpy() == 1, "Man", "Woman")
        premier = c.premier.to_numpy() == 1
        h = mischen(c["hash(customerId)"].to_numpy())
        a = np.full(len(c), -1, dtype=np.int16)
        b = np.full(len(c), -1, dtype=np.int16)
        for (zalter, zgeschlecht), idx in zellen.items():
            m = (alter == zalter) & (geschlecht == zgeschlecht)
            idx = np.array(idx)
            treu = np.array([i for i in idx if personas[i]["attribute"]["lstyle_shopping_style"] == "Brand loyal"
                             or personas[i]["attribute"]["economic_motivation"] == "Value-driven"]) if len(idx) else idx
            a[m] = idx[(h[m] % np.uint64(len(idx))).astype(np.int64)]
            b[m] = a[m]
            mp = m & premier
            if len(treu):
                b[mp] = treu[(h[mp] % np.uint64(len(treu))).astype(np.int64)]
        assert (a >= 0).all() and (b >= 0).all()
        teile.append(pd.DataFrame({"split": split, "kunde": c["hash(customerId)"].to_numpy(), "alter": alter,
                                   "geschlecht": geschlecht, "premier": premier, "persona_A": a, "persona_B": b}))
    df = pd.concat(teile, ignore_index=True)
    df.to_parquet(DATEN / "kunden_persona.parquet", index=False)

    beide = df.pivot_table(index="kunde", columns="split", values="persona_A", aggfunc="first").dropna()
    gleich = (beide.training == beide.testing).mean() if len(beide) else float("nan")
    print(f"{len(df)} Zeilen -> {DATEN / 'kunden_persona.parquet'}; Kund:innen in beiden Splits: {len(beide)}, "
          f"gleiche Persona: {gleich:.2%}")
    print("Zellen (Training):", df[df.split == "training"].groupby(["alter", "geschlecht"]).size()
          .div((df.split == "training").sum()).round(3).to_dict())


def kontext():
    kp = pd.read_parquet(DATEN / "kunden_persona.parquet")
    grenzen = None
    for split in ("training", "testing"):
        ev = pd.read_pickle(ASOS / f"event_table_{split}.p")
        pr = pd.read_pickle(ASOS / f"product_nodes_{split}.p")[
            ["hash(variantID)", "avgGbpPrice", "avgDiscountValue"]]
        assert pr["hash(variantID)"].is_unique
        n = len(ev)
        ev = ev.merge(pr, on="hash(variantID)", how="left")
        assert len(ev) == n
        preis = ev.avgGbpPrice.to_numpy()
        rabatt = (ev.avgDiscountValue / (ev.avgGbpPrice + ev.avgDiscountValue)).to_numpy()
        if grenzen is None:
            grenzen = {"preis": np.nanquantile(preis, [1 / 3, 2 / 3]), "sale": float(np.nanmedian(rabatt))}
        preisstufe = np.digitize(preis, grenzen["preis"])
        sale = (rabatt >= grenzen["sale"]).astype(np.int64)
        teil = kp[kp.split == split].set_index("kunde")
        codes = {}
        for variante in ("A", "B"):
            persona = ev["hash(customerId)"].map(teil[f"persona_{variante}"]).to_numpy(dtype=float)
            ok = ~(np.isnan(persona) | np.isnan(preis) | np.isnan(rabatt))
            code = np.full(n, -1, dtype=np.int64)
            code[ok] = kontext_code(persona[ok].astype(np.int64), preisstufe[ok], sale[ok])
            codes[variante] = code
        np.savez(DATEN / f"kontext_{split}.npz", **codes, preis_grenzen=grenzen["preis"], sale_grenze=grenzen["sale"])
        print(f"{split}: {n} Käufe, ohne Situation {np.mean(codes['A'] < 0):.2%} | Preisstufen "
              f"{np.bincount(preisstufe[ok], minlength=3) / ok.sum()} | Sale {sale[ok].mean():.3f}")
    print(f"Grenzen aus dem Training: Preis {np.round(grenzen['preis'], 2)} GBP, Rabattanteil {grenzen['sale']:.3f}")


@lru_cache
def preis_texte():
    a, b = np.round(np.load(DATEN / "kontext_training.npz")["preis_grenzen"]).astype(int)
    return [f"under £{a}", f"between £{a} and £{b}", f"over £{b}"]


def stufen_text():
    return "\n".join(f"{s} = {STUFEN_TEXT[s]} (chance that I remove the item: {round(p * 100)} %)."
                     for s, p in sorted(STUFEN_P.items()))


@lru_cache
def prompt_version():
    teile = [SYSTEM, SYSTEM_OHNE_PERSONA, USER, SALE_TEXT, ZUSTAND_TEXT, WARNUNG_TEXT, stufen_text(), preis_texte(),
             list(kohorte()["labels"].items()), [p["persona_id"] for p in kohorte()["personas"]]]
    return hashlib.sha256(json.dumps(teile, ensure_ascii=False).encode()).hexdigest()[:10]


def system_prompt(attribute):
    if attribute is None:
        return SYSTEM_OHNE_PERSONA
    zeilen = [f"- {label}: {attribute[aid]}" for aid, label in kohorte()["labels"].items()]
    return SYSTEM + "\n\nPersona:\n" + "\n".join(zeilen)


def user_prompt(preis, sale, zustand, warnung):
    return USER.format(preis=preis_texte()[preis], sale=SALE_TEXT[sale], zustand=ZUSTAND_TEXT[zustand],
                       warnung=WARNUNG_TEXT[warnung], stufen=stufen_text())


def anfragen(aufgabe):
    personas = kohorte()["personas"]
    situationen = [(preis, sale, zustand) for preis in range(len(PREISE)) for sale in range(len(SALE))
                   for zustand in range(len(ZUSTAENDE))]
    if aufgabe == "haupt":   # feste Zufallsreihenfolge, damit ein begrenzter Lauf (Pilot) über alle Zellen streut
        faelle = [(personas[i]["index"], str(personas[i]["index"]), personas[i]["attribute"], {}, situationen)
                  for i in np.random.default_rng(SEED).permutation(len(personas))]
    elif aufgabe == "treue":
        basis = np.sort(np.random.default_rng(SEED).choice(len(personas), TREUE_BASIS, replace=False))
        faelle = [(None, f"t{ai}{arm}x{b}", {**personas[b]["attribute"], aid: wert},
                   {"attribut": aid, "arm": ["hoch", "niedrig"][arm], "basis": int(b)}, TREUE_SITUATIONEN)
                  for ai, (aid, werte) in enumerate(TREUE.items()) for arm, wert in enumerate(werte) for b in basis]
    elif aufgabe == "ohne_persona":
        faelle = [(None, "x", None, {}, situationen)]
    else:
        raise ValueError(f"unbekannte Aufgabe {aufgabe}")
    for persona, kennung, attribute, extra, sits in faelle:
        system = system_prompt(attribute)
        for preis, sale, zustand in sits:
            for warnung in AKTIVE_WARNUNGEN:
                w = WARNUNGEN.index(warnung)
                yield {"aufgabe": aufgabe, "persona": persona, "kennung": kennung, **extra, "preis": preis,
                       "sale": sale, "zustand": zustand, "warnung": warnung,
                       "schluessel": f"{aufgabe[0]}-{kennung}-{preis}{sale}{zustand}{w}",
                       "system": system, "user": user_prompt(preis, sale, zustand, warnung)}


def offene_anfragen(aufgabe, modell=MODELL):
    k_n = K_OHNE_PERSONA if aufgabe == "ohne_persona" else K
    df = lade_puffer(PUFFER)
    ok, versuche = set(), {}
    if not df.empty:
        df = df[(df.aufgabe == aufgabe) & (df.modell == modell) & (df.prompt_version == prompt_version())]
        for schluessel, k, status in zip(df.schluessel, df.k, df.status):
            versuche[(schluessel, k)] = versuche.get((schluessel, k), 0) + 1
            if status == "ok":
                ok.add((schluessel, k))
    for anfrage in anfragen(aufgabe):
        for k in range(k_n):
            v = versuche.get((anfrage["schluessel"], k), 0)
            if (anfrage["schluessel"], k) not in ok and v < MAX_VERSUCHE:
                yield {**anfrage, "k": k, "versuch": v + 1, "custom_id": f"{anfrage['schluessel']}-k{k}-v{v + 1}",
                       "schema": SCHEMA}


def prompts(aufgabe="haupt", modell=MODELL):
    liste = list(anfragen(aufgabe))
    k_n = K_OHNE_PERSONA if aufgabe == "ohne_persona" else K
    offen = sum(1 for _ in offene_anfragen(aufgabe, modell))
    zeichen_s = np.mean([len(a["system"]) for a in liste])
    zeichen_u = np.mean([len(a["user"]) for a in liste])
    print(f"Aufgabe {aufgabe}: {len(liste)} Schlüssel x {k_n} = {len(liste) * k_n} Anfragen, offen für {modell}: "
          f"{offen}, PROMPT_VERSION {prompt_version()}")
    print(f"Länge im Mittel: system {zeichen_s:.0f} Zeichen (~{zeichen_s / 4:.0f} Token), user {zeichen_u:.0f} "
          f"Zeichen (~{zeichen_u / 4:.0f} Token); verschiedene system-Prompts: {len({a['system'] for a in liste})}")
    print("\n[system]\n" + liste[0]["system"] + "\n\n[user]\n" + liste[0]["user"])


def jetzt():
    return datetime.now().isoformat(timespec="seconds")


@lru_cache
def modelle():
    r = httpx.get(f"{OPENROUTER}/models", timeout=60)
    r.raise_for_status()
    return {m["id"]: m for m in r.json()["data"]}


def zerlegen(modell):
    slug, _, anbieter = modell.partition("@")
    return slug, anbieter or None


def provider(modell):
    anbieter = zerlegen(modell)[1]
    return {"only": [anbieter]} if anbieter else {}


def mit_batch(modell):
    return f"{zerlegen(modell)[0]}:batch" in modelle()


def modell_info(modell):
    slug = zerlegen(modell)[0]
    info = modelle().get(f"{slug}:batch") or modelle().get(slug)
    if info is None:
        raise SystemExit(f"{slug} gibt es bei OpenRouter nicht. Passende Modelle: python Puffer.py modelle")
    if "structured_outputs" not in info["supported_parameters"]:
        raise SystemExit(f"{slug} kann kein JSON-Schema (structured_outputs). Passende Modelle: python Puffer.py modelle")
    return info


def denken(info):
    r = info.get("reasoning")
    if r is None or "reasoning" not in info["supported_parameters"]:
        return None
    stufen = r.get("supported_efforts") or []
    if "none" in stufen:
        return {"effort": "none"}
    if not r.get("mandatory"):
        return {"enabled": False}
    return {"effort": min((s for s in stufen if s in STUFEN_DENKEN), key=STUFEN_DENKEN.index, default="low"),
            "exclude": True}


def parameter(anfrage, modell=MODELL):
    body = {"messages": [{"role": "system", "content": anfrage["system"]}, {"role": "user", "content": anfrage["user"]}],
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "antwort", "strict": True, "schema": anfrage["schema"]}}}
    if (r := denken(modell_info(modell))) is not None:
        body["reasoning"] = r
    return {**body, **MODELL_PARAMETER.get(zerlegen(modell)[0], {})}


def api():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Kein API-Key gefunden. Key unter openrouter.ai/settings/keys anlegen, in ~/.bashrc die Zeile\n"
                         "  export OPENROUTER_API_KEY=\"sk-or-...\"\nergänzen und ein neues Terminal öffnen.")
    return httpx.Client(base_url=OPENROUTER, headers={"Authorization": f"Bearer {key}"}, timeout=600)


def aufrufen(client, methode, pfad, versuche=5, einzeln=False, **kwargs):
    for _ in range(versuche):
        try:
            r = client.request(methode, pfad, **kwargs)
        except httpx.TransportError as e:
            fehler = f"keine Verbindung ({e})"
        else:
            if r.is_success:
                return r.json()
            try:
                meldung = r.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                meldung = r.text[:300]
            fehler = f"{r.status_code}: {meldung}"
            # einen gerade angelegten Batch kennt OpenRouter erst nach einigen Sekunden
            neu = r.status_code == 404 and methode == "GET" and pfad.startswith("/batches/")
            if r.status_code != 429 and r.status_code < 500 and not neu:
                if einzeln and r.status_code not in (401, 402):
                    return {"error": {"code": r.status_code, "message": meldung}}
                hinweis = {401: " (OPENROUTER_API_KEY prüfen)", 402: " (Guthaben unter openrouter.ai/settings/credits)",
                           410: f" (Batch älter als 30 Tage, seine Zeile in {BATCHES} löschen)"}.get(r.status_code, "")
                raise SystemExit(f"OpenRouter-Fehler {fehler}{hinweis}")
        print(f"{jetzt()} OpenRouter {fehler}", flush=True)
        if versuche > 1:
            time.sleep(WARTEN)
    if einzeln:
        return {"error": {"code": "wiederholt", "message": fehler}}
    raise SystemExit(f"OpenRouter-Fehler {fehler}" + (f" nach {versuche} Versuchen" if versuche > 1 else ""))


def auswerten(body):
    wahl = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    a = {"finish_reason": wahl.get("finish_reason"), "modell_antwort": body.get("model"),
         "anbieter": body.get("provider"), "generation_id": body.get("id"), "input_tokens": usage.get("prompt_tokens"),
         "output_tokens": usage.get("completion_tokens"),
         "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
         "kosten": usage.get("cost"), "response": None, "fehler": None}
    if body.get("error"):
        a["fehler"] = f"{body['error'].get('code')}: {body['error'].get('message')}"
    elif a["finish_reason"] != "stop":
        a["fehler"] = f"finish_reason {a['finish_reason']} ({wahl.get('native_finish_reason')})"
    else:
        text = ((wahl.get("message") or {}).get("content") or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            a["response"] = json.loads(text)["response"]
        except (ValueError, KeyError, TypeError):
            a["fehler"] = f"kein gültiges JSON: {text[:200]}"
    return a


def senden(anfragen, modell=MODELL):
    client = api()
    for start in range(0, len(anfragen), BATCH_GROESSE):
        teil = anfragen[start:start + BATCH_GROESSE]
        # endpoint, model und provider müssen vor requests stehen
        kopf = {"endpoint": "/v1/chat/completions", "model": zerlegen(modell)[0],
                **({"provider": provider(modell)} if provider(modell) else {})}
        # versuche=1: eine Wiederholung könnte einen zweiten, ebenfalls bezahlten Batch anlegen
        batch = aufrufen(client, "POST", "/batches", versuche=1,
                         json={**kopf, "requests": [{"custom_id": a["custom_id"], "body": parameter(a, modell)}
                                                    for a in teil]})
        anhaengen([{"batch_id": batch["id"], "aufgabe": teil[0]["aufgabe"], "modell": modell,
                    "prompt_version": prompt_version(), "anfragen": len(teil), "erstellt": jetzt()}], BATCHES)
        print(f"Batch {batch['id']} mit {len(teil)} Anfragen angelegt", flush=True)


def offene_batches():
    if not BATCHES.exists():
        return []
    zeilen = [json.loads(z) for z in BATCHES.read_text().splitlines() if z.strip()]
    abgeholt = {z["batch_id"] for z in zeilen if "abgeholt" in z}
    return [z for z in zeilen if "erstellt" in z and z["batch_id"] not in abgeholt]


def abholen():
    client = api()
    for b in offene_batches():
        if b["prompt_version"] != prompt_version():
            raise SystemExit(f"Batch {b['batch_id']} gehört zu PROMPT_VERSION {b['prompt_version']}, die aktuelle ist "
                             f"{prompt_version()}. Prompt zurücksetzen oder die Zeile des Batches in {BATCHES} löschen.")
        try:
            while (batch := aufrufen(client, "GET", f"/batches/{b['batch_id']}"))["status"] not in (
                    "completed", "failed", "expired", "cancelled"):
                n = batch["request_counts"]
                print(f"{jetzt()} {b['batch_id']}: {batch['status']}, {n['completed']} fertig, {n['failed']} Fehler "
                      f"von {n['total']}", flush=True)
                time.sleep(WARTEN)
        except KeyboardInterrupt:
            raise SystemExit("\nWarten abgebrochen, der Batch läuft weiter. Erneut python Puffer.py fuellen holt die "
                             "Ergebnisse ab.")
        offen = {a["custom_id"]: a for a in offene_anfragen(b["aufgabe"], b["modell"])}
        zeilen, doppelt = [], 0
        for r in batch.get("results") or []:
            if r["custom_id"] not in offen:
                doppelt += 1
                continue
            antwort = r.get("response") or {}
            if r.get("error") is None and antwort.get("status_code") == 200:
                antwort = auswerten(antwort["body"])
            else:
                e = r.get("error") or (antwort.get("body") or {}).get("error") or {}
                antwort = {"fehler": f"{e.get('error_type') or e.get('type') or e.get('code')}: {e.get('message')}"}
            zeilen.append(pruefen(offen[r["custom_id"]], {**antwort, "batch_id": b["batch_id"]}, b["modell"]))
        anhaengen(zeilen)
        kosten = (batch.get("usage") or {}).get("cost")
        anhaengen([{"batch_id": b["batch_id"], "abgeholt": jetzt(), "status": batch["status"], "kosten": kosten}],
                  BATCHES)
        if batch["status"] != "completed":
            print(f"{b['batch_id']}: {batch['status']} ohne Ergebnisse ({(batch.get('error') or {}).get('message')}), "
                  "die Anfragen bleiben offen", flush=True)
            continue
        print(f"{b['batch_id']}: gültig {sum(z['status'] == 'ok' for z in zeilen)} von {len(zeilen)} -> {PUFFER}, "
              f"Kosten {'?' if kosten is None else f'{kosten:.2f}'} $"
              + (f", {doppelt} standen schon im Puffer" if doppelt else ""), flush=True)


def direkt_body(anfrage, modell=MODELL):
    return {"model": zerlegen(modell)[0], **parameter(anfrage, modell),
            "provider": {"require_parameters": True, **provider(modell)}}


def direkt(anfragen, modell=MODELL):
    client = api()

    def fragen(a):
        return a, auswerten(aufrufen(client, "POST", "/chat/completions", einzeln=True, json=direkt_body(a, modell)))

    pool = ThreadPoolExecutor(PARALLEL)
    futures = [pool.submit(fragen, a) for a in anfragen]
    gespeichert = set()
    gueltig = in_folge = 0

    def speichern(future):
        nonlocal gueltig, in_folge
        zeile = pruefen(*future.result(), modell)
        anhaengen([zeile])
        gespeichert.add(future)
        gueltig += zeile["status"] == "ok"
        in_folge = 0 if zeile["status"] == "ok" else in_folge + 1

    try:
        for future in as_completed(futures):
            speichern(future)
            if in_folge >= FEHLER_IN_FOLGE:
                raise SystemExit(f"{in_folge} Fehler hintereinander, zuletzt: {pruefen(*future.result(), modell)['fehler']}")
            if len(gespeichert) % 500 == 0 or len(gespeichert) == len(anfragen):
                print(f"{jetzt()} {len(gespeichert)} von {len(anfragen)} beantwortet, gültig {gueltig} -> {PUFFER}",
                      flush=True)
    except (KeyboardInterrupt, SystemExit) as e:
        pool.shutdown(wait=False, cancel_futures=True)
        for future in futures:   # bereits beantwortete (und bezahlte) Anfragen noch speichern
            if future.done() and not future.cancelled() and future.exception() is None and future not in gespeichert:
                speichern(future)
        raise SystemExit(f"{e or 'Abgebrochen'}\n{len(gespeichert)} von {len(anfragen)} Antworten stehen im Puffer. "
                         "Erneut python Puffer.py fuellen macht mit den offenen weiter.")
    pool.shutdown()


def probe(aufgabe="haupt", modell=MODELL, n=3):
    offen = list(offene_anfragen(aufgabe, modell))
    if not offen:
        print(f"keine offenen Anfragen ({aufgabe}, {modell})")
        return
    preise = modell_info(modell)["pricing"]
    client = api()
    token, kosten_je = [], []
    for a in offen[::max(1, len(offen) // n)][:n]:
        w = auswerten(aufrufen(client, "POST", "/chat/completions", json=direkt_body(a, modell)))
        token.append((w["input_tokens"] or 0, w["output_tokens"] or 0))
        kosten_je.append(w["kosten"])
        print(f"{a['custom_id']}: Stufe {w['response']}, finish_reason {w['finish_reason']}, Fehler {w['fehler']}, "
              f"Anbieter {w['anbieter']}, Token ein {w['input_tokens']}, aus {w['output_tokens']} (davon Denken "
              f"{w['reasoning_tokens']}), Kosten {w['kosten']} $", flush=True)
    ein, aus = np.mean(token, axis=0)
    kosten = len(offen) * (np.mean(kosten_je) if not mit_batch(modell) and None not in kosten_je
                           else ein * float(preise["prompt"]) + aus * float(preise["completion"]))
    print(f"Schätzung für alle {len(offen)} offenen Anfragen ({aufgabe}, {modell}) "
          f"{'im Batch' if mit_batch(modell) else 'direkt'}: etwa {kosten:.2f} $")


def modelle_zeigen(suche=""):
    n = sum(1 for _ in anfragen("haupt")) * K
    zeilen = []
    for slug in modelle():
        if ":" in slug or slug.startswith("~") or suche not in slug:
            continue
        info = modelle().get(f"{slug}:batch") or modelle()[slug]
        if "structured_outputs" in info["supported_parameters"]:
            ein, aus = float(info["pricing"]["prompt"]), float(info["pricing"]["completion"])
            zeilen.append((n * (500 * ein + 10 * aus), slug, ein * 1e6, aus * 1e6, mit_batch(slug), denken(info)))
    print(f"Preise in $ je Mio. Token (Eingabe / Ausgabe), mit Batch die Batch-Preise; Hauptlauf {n} Anfragen ohne "
          "Denk-Token")
    for kosten, slug, ein, aus, batch, r in sorted(zeilen):
        print(f"{slug:45s} {'Batch ' if batch else 'direkt'} {ein:8.3f} / {aus:8.3f}   Hauptlauf ab {kosten:7.2f} $   "
              "Denken: " + ("aus" if r is None or r.get("effort") == "none" or r.get("enabled") is False
                            else f"immer, Stufe {r['effort']} (kostet zusätzlich)"))


def pruefen(anfrage, antwort, modell=MODELL):
    stufe = antwort.get("response")
    ok = (antwort.get("fehler") is None and antwort.get("finish_reason") == "stop"
          and isinstance(stufe, int) and not isinstance(stufe, bool) and stufe in STUFEN_P)
    zeile = {k: anfrage.get(k) for k in ("custom_id", "schluessel", "aufgabe", "persona", "kennung", "attribut", "arm",
                                         "basis", "preis", "sale", "zustand", "warnung", "k", "versuch")}
    zeile.update({"modell": modell, "prompt_version": prompt_version(),
                  "status": "ok" if ok else "fehler", "response": stufe if ok else None,
                  "finish_reason": antwort.get("finish_reason"), "fehler": antwort.get("fehler"),
                  **{k: antwort.get(k) for k in ("modell_antwort", "anbieter", "input_tokens", "output_tokens",
                                                  "reasoning_tokens", "kosten", "generation_id", "batch_id")},
                  "zeit": jetzt()})
    return zeile


def anhaengen(zeilen, pfad=None):
    with open(pfad or PUFFER, "a", encoding="utf-8") as f:
        for z in zeilen:
            f.write(json.dumps(z, ensure_ascii=False) + "\n")


def fuellen(aufgabe="haupt", modell=MODELL, anzahl=None):
    if offene_batches():
        print("Es gibt noch nicht abgeholte Batches, diese zuerst abholen", flush=True)
        return abholen()
    batch = mit_batch(modell)
    offen = list(offene_anfragen(aufgabe, modell))[:anzahl]
    print(f"{len(offen)} offene Anfragen ({aufgabe}, {modell}, PROMPT_VERSION {prompt_version()}, "
          f"{'als Batch' if batch else f'direkt, {PARALLEL} gleichzeitig'})", flush=True)
    if not offen:
        return
    if batch:
        senden(offen, modell)
        abholen()
    else:
        direkt(offen, modell)


def benjamini_hochberg(p):
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    ordnung = np.argsort(p)
    q = p[ordnung] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(max=1)
    return q[np.argsort(ordnung)]


def batch_kosten():
    if not BATCHES.exists():
        return {}
    zeilen = [json.loads(z) for z in BATCHES.read_text().splitlines() if z.strip()]
    kosten = {z["batch_id"]: z.get("kosten") or 0.0 for z in zeilen if "abgeholt" in z}
    summe = {}
    for z in zeilen:
        if "erstellt" in z and z["batch_id"] in kosten:
            schluessel = (z["aufgabe"], z["modell"], z["prompt_version"])
            summe[schluessel] = summe.get(schluessel, 0.0) + kosten[z["batch_id"]]
    return summe


def bericht():
    df = lade_puffer(PUFFER)
    if df.empty:
        print(f"Puffer {PUFFER} ist leer")
        return
    personas = pd.DataFrame([{**p["attribute"], **p["zelle"], "persona": p["index"]} for p in kohorte()["personas"]])
    for (aufgabe, modell, version), g in df.groupby(["aufgabe", "modell", "prompt_version"]):
        k_n = K_OHNE_PERSONA if aufgabe == "ohne_persona" else K
        gueltig = g[g.status == "ok"]
        voll = (gueltig.groupby("schluessel").k.nunique() >= k_n).sum()
        print(f"\n== {aufgabe} | {modell} | PROMPT_VERSION {version}{'' if version == prompt_version() else ' (alt)'}")
        print(f"Zeilen {len(g)}, gültig {len(gueltig)}, Fehler {len(g) - len(gueltig)}; Schlüssel mit {k_n} gültigen "
              f"Antworten: {voll} von {sum(1 for _ in anfragen(aufgabe))}")
        token = {k: g[k].fillna(0).sum() for k in ("input_tokens", "output_tokens", "reasoning_tokens")}
        kosten = batch_kosten().get((aufgabe, modell, version), 0.0) + g.kosten[g.batch_id.isna()].fillna(0).sum()
        print("Token: " + ", ".join(f"{k} {v:.0f}" for k, v in token.items()) + " | Kosten laut OpenRouter "
              f"{kosten:.2f} $ | Anbieter: "
              f"{g.anbieter.fillna('?').value_counts().to_dict()}")
        if gueltig.empty:
            continue
        gueltig = gueltig.assign(response=gueltig.response.astype(int), p=gueltig.response.astype(int).map(STUFEN_P))
        print("Stufen:", gueltig.response.value_counts(normalize=True).sort_index().round(3).to_dict(),
              "| Anteil Mitte (3):", round((gueltig.response == 3).mean(), 3),
              "| Schlüssel mit lauter gleichen Antworten:", round((gueltig.groupby("schluessel").response.nunique() == 1).mean(), 3))
        if aufgabe == "haupt":
            h = gueltig.assign(persona=gueltig.persona.astype(int)).merge(personas, on="persona")
            haeufig = h.groupby("persona").response.value_counts(normalize=True).unstack(fill_value=0)
            print("Entropie je Persona (Mittel, max. log 5 = 1.61):",
                  round(float(-(haeufig * np.log(haeufig.where(haeufig > 0, 1))).sum(axis=1).mean()), 3))
            for spalte, namen in [("zustand", ZUSTAENDE), ("preis", PREISE), ("sale", SALE)]:
                print(f"P(Abbruch) nach {spalte}:", {namen[i]: round(v, 3) for i, v in h.groupby(spalte).p.mean().items()})
            p_kein, p_grund = h.groupby("zustand").p.mean().reindex([0, 1])
            print("Abbruch nach Warnung wie im Simulator (stochastisch 0,5 / 0,2): Retoure",
                  round(P_GRUND_RETOURE * p_grund + (1 - P_GRUND_RETOURE) * p_kein, 3), "| behalten",
                  round(P_GRUND_BEHALTEN * p_grund + (1 - P_GRUND_BEHALTEN) * p_kein, 3))
            for spalte in ("alter", "geschlecht", "lstyle_shopping_style", "economic_motivation", "trust_level"):
                print(f"P(Abbruch) nach {spalte}:", h.groupby(spalte).p.mean().round(3).to_dict())
        elif aufgabe == "treue":
            from scipy.stats import wilcoxon
            zeilen = []
            for aid, t in gueltig.groupby("attribut"):
                # gepaart je Basis-Persona und Situation: Mittel über die K Antworten, Arm hoch gegen Arm niedrig
                paare = (t.groupby(["basis", "preis", "sale", "zustand", "arm"]).p.mean().unstack("arm")
                         .reindex(columns=["hoch", "niedrig"]).dropna())
                d = paare["hoch"] - paare["niedrig"]
                zeilen.append((aid, paare["hoch"].mean(), paare["niedrig"].mean(), len(d),
                               wilcoxon(d).pvalue if (d != 0).any() else 1.0))
            for (aid, ph, pn, n, _), qi in zip(zeilen, benjamini_hochberg([z[-1] for z in zeilen])):
                print(f"Treue {aid:24s} P(Abbruch) {TREUE[aid][0]} {ph:.3f} vs {TREUE[aid][1]} {pn:.3f} "
                      f"({n} Paare), q = {qi:.3f}")
        elif aufgabe == "ohne_persona":
            print("P(Abbruch) ohne Persona nach Zustand:",
                  {ZUSTAENDE[i]: round(v, 3) for i, v in gueltig.groupby("zustand").p.mean().items()})


if __name__ == "__main__":
    befehle = ["zuordnung", "kontext", "prompts", "modelle", "probe", "fuellen", "bericht"]
    parser = argparse.ArgumentParser(description="LLM-Nutzersimulation: Puffer über OpenRouter füllen")
    parser.add_argument("befehl", nargs="?", default="prompts", choices=befehle)
    parser.add_argument("aufgabe", nargs="?", help="haupt (Standard), treue oder ohne_persona; bei modelle: Suchtext")
    parser.add_argument("anzahl", nargs="?", type=int, help="höchstens so viele Anfragen (fuellen, etwa im Pilot)")
    parser.add_argument("--modell", default=MODELL, help=f"OpenRouter-Slug, Standard {MODELL}")
    parser.add_argument("--anbieter", help="nur dieser OpenRouter-Anbieter, etwa deepinfra/bf16 (Endpunkte des Modells: "
                                           "openrouter.ai/<Slug>/providers)")
    a = parser.parse_args()
    if a.anbieter:
        a.modell = f"{zerlegen(a.modell)[0]}@{a.anbieter}"
    aufgabe = a.aufgabe or "haupt"
    try:
        {"zuordnung": zuordnung, "kontext": kontext, "bericht": bericht,
         "prompts": lambda: prompts(aufgabe, a.modell), "modelle": lambda: modelle_zeigen(a.aufgabe or ""),
         "probe": lambda: probe(aufgabe, a.modell), "fuellen": lambda: fuellen(aufgabe, a.modell, a.anzahl)}[a.befehl]()
    except httpx.HTTPError as e:
        raise SystemExit(f"OpenRouter nicht erreichbar: {e}")
