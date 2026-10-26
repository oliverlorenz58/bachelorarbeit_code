import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATEN = Path(__file__).resolve().parent / "daten"
REPO = "MatrAIx2026/MatrAIx_Persona_1M"
REVISION = "8b1073ab23d0c0ba0928386a041bac55e5365ddc"
SHARDS = [f"data/persona-1m-{i:04d}.parquet" for i in range(6, 10)]   # die synthetischen Personas
METADATEN = ["manifest.json", "persona_codes.schema.json"]

# Attribute im Persona-Block; sensible Attribute und die Region fehlen bewusst
ATTRIBUTE = [
    "age_bracket", "gender_identity", "urbanicity",
    "life_stage", "demo_employment_status", "socioeconomic_band", "lstyle_household_size", "demo_parental_status",
    "lstyle_shopping_style", "economic_motivation", "att_consumerism", "att_brand_loyalty", "att_online_reviews",
    "pref_quality_vs_quantity", "pref_novelty_vs_familiarity", "skill_budgeting", "topic_fashion", "tech_savviness",
    "risk_tolerance", "decision_style", "need_for_closure", "time_pressure", "trust_level",
    "bfi2_domain_conscientiousness", "bfi2_facet_anxiety",
    "val_sustainability", "val_social_status", "schwartz_value_hedonism",
]

ALTERSSTUFEN = {"18-24": "18-24", "25-34": "25-34", "35-44": "35-44", "45-54": "45-54",
                "55-64": "55+", "65-74": "55+", "75-84": "55+", "85+": "55+"}
GESCHLECHTER = ["Woman", "Man"]
NICHT_PLAUSIBEL = {"urbanicity": ["Nomadic / remote"], "tech_savviness": ["Avoidant"]}
DOKUMENTATION = ["region"]
EINKAUFSSTILE = ["Bargain hunter", "Researcher", "Minimalist", "Brand loyal", "Impulse buyer"]
MOTIVATIONEN = ["Cost-sensitive", "Value-driven", "Indifferent", "Premium-seeking"]
N_JE_KOMBINATION = 2
SEED = 2026


def laden():
    from huggingface_hub import hf_hub_download
    for f in METADATEN + SHARDS:
        hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=DATEN)
    summen = {f["path"]: f["sha256"] for f in json.loads((DATEN / "manifest.json").read_text())["files"]}
    for f in SHARDS:
        assert hashlib.sha256((DATEN / f).read_bytes()).hexdigest() == summen[f], f"Prüfsumme falsch: {f}"
    print("geladen und geprüft:", ", ".join(SHARDS))


def dekodieren(pfad, schema, ids):
    t = pq.read_table(pfad, columns=["source", "source_record_id", "source_row_index", "attributes", "null_bitmap",
                                     "attribute_overrides"])
    n = t.num_rows
    codes = np.frombuffer(b"".join(t.column("attributes").to_pylist()), dtype=np.uint8).reshape(n, -1)
    nulls = np.zeros((n, (len(schema) + 7) // 8), dtype=np.uint8)
    for zeile, bitmap in enumerate(t.column("null_bitmap").to_pylist()):
        if bitmap is not None:
            nulls[zeile] = np.frombuffer(bitmap, dtype=np.uint8)

    index = {c["id"]: i for i, c in enumerate(schema)}
    ids_quelle = t.column("source_record_id").to_pylist()
    out = {"persona_id": [rid or f"{src}-{row}" for rid, src, row in
                          zip(ids_quelle, t.column("source").to_pylist(), t.column("source_row_index").to_pylist())]}
    for aid in ids:
        i = index[aid]
        code = codes[:, i // 2] & 0x0F if i % 2 == 0 else codes[:, i // 2] >> 4
        werte = np.array(schema[i]["values"] + [None] * 16, dtype=object)[code]
        werte[(nulls[:, i // 8] >> (i % 8)) & 1 == 1] = None
        out[aid] = werte
    for zeile, overrides in enumerate(t.column("attribute_overrides").to_pylist()):
        for o in overrides or []:
            aid = schema[o["field_index"]]["id"]
            if aid in out:
                out[aid][zeile] = o["value"]
    df = pd.DataFrame(out)
    return df.mask(df == "null")


def pool(schema):
    df = pd.concat([dekodieren(DATEN / f, schema, ATTRIBUTE + DOKUMENTATION) for f in SHARDS], ignore_index=True)
    assert df.persona_id.is_unique
    df["alter"] = df.age_bracket.map(ALTERSSTUFEN)
    ok = (df.alter.notna() & df.gender_identity.isin(GESCHLECHTER)
          & df.lstyle_shopping_style.isin(EINKAUFSSTILE) & df.economic_motivation.isin(MOTIVATIONEN)
          & df[ATTRIBUTE].notna().all(axis=1))
    for aid, werte in NICHT_PLAUSIBEL.items():
        ok &= ~df[aid].isin(werte)
    print(f"{len(df)} synthetische Personas, {ok.sum()} erfüllen die Filter", flush=True)
    return df[ok].sort_values("persona_id").reset_index(drop=True)


def ziehen():
    schema = json.loads((DATEN / "persona_codes.schema.json").read_text())["columns"]
    labels = {c["id"]: c["label"] for c in schema}
    kandidaten = pool(schema)
    rng = np.random.default_rng(SEED)
    personas, strata = [], []
    for alter in sorted(set(ALTERSSTUFEN.values())):
        for geschlecht in GESCHLECHTER:
            for stil in EINKAUFSSTILE:
                for motivation in MOTIVATIONEN:
                    k = kandidaten[(kandidaten.alter == alter) & (kandidaten.gender_identity == geschlecht)
                                   & (kandidaten.lstyle_shopping_style == stil)
                                   & (kandidaten.economic_motivation == motivation)]
                    n = min(N_JE_KOMBINATION, len(k))
                    for _, p in k.iloc[np.sort(rng.choice(len(k), size=n, replace=False))].iterrows():
                        personas.append({"index": len(personas), "persona_id": p.persona_id,
                                         "zelle": {"alter": alter, "geschlecht": geschlecht},
                                         "attribute": {aid: p[aid] for aid in ATTRIBUTE},
                                         "dokumentation": {aid: p[aid] for aid in DOKUMENTATION}})
                    strata.append({"alter": alter, "geschlecht": geschlecht, "einkaufsstil": stil,
                                   "motivation": motivation, "pool": len(k), "gezogen": n})

    summen = {f["path"]: f["sha256"] for f in json.loads((DATEN / "manifest.json").read_text())["files"]}
    kohorte = {
        "manifest": {
            "erstellt": datetime.now().isoformat(timespec="seconds"),
            "quelle": {"datensatz": REPO, "revision": REVISION, "shards": {f: summen[f] for f in SHARDS}},
            "angefragt": {
                "filter": {"source": "synthetic", "age_bracket": ALTERSSTUFEN, "gender_identity": GESCHLECHTER,
                           "lstyle_shopping_style": EINKAUFSSTILE, "economic_motivation": MOTIVATIONEN,
                           "ausgeschlossen": NICHT_PLAUSIBEL, "alle_attribute_befuellt": True,
                           "region": "alle (nicht im Persona-Block)"},
                "strata": ["alter", "geschlecht", "einkaufsstil", "motivation"],
                "n_je_kombination": N_JE_KOMBINATION, "seed": SEED, "attribute": ATTRIBUTE,
            },
            "realisiert": {"pool": len(kandidaten), "personas": len(personas),
                           "unterbesetzt": [s for s in strata if s["gezogen"] < N_JE_KOMBINATION], "strata": strata},
        },
        "labels": {aid: labels[aid] for aid in ATTRIBUTE},
        "personas": personas,
    }
    (DATEN / "kohorte.json").write_text(json.dumps(kohorte, indent=1, ensure_ascii=False))
    print(f"{len(personas)} Personas in {len(strata)} Strata -> {DATEN / 'kohorte.json'}, "
          f"unterbesetzt: {len(kohorte['manifest']['realisiert']['unterbesetzt'])}")


if __name__ == "__main__":
    laden() if len(sys.argv) > 1 and sys.argv[1] == "laden" else ziehen()
