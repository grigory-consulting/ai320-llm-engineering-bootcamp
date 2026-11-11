"""Lab 4: vorbereitete Läufe (Trainer, vor dem Kurs). Aufruf im Ordner labs/:

    python -m common.lab4_prepare e5 tfidf                    # CPU: Embeddings wie im Ersatzweg der Labs, Baselines
    LAB4_DEVICE=mps python -m common.lab4_prepare main kopf cv   # Fine-Tuning-Stufen auf der Apple-GPU, rund 1 h
    python -m common.lab4_prepare kopf_lr zwillinge klein kurz klein_cv inferenz   # CPU, rund 75 min
    python -m common.lab4_prepare klein_hebel                 # CPU, rund 7 min

Die Reihenfolge zählt: inferenz braucht das Ergebnis von tfidf, kurz das von main. Ohne LAB4_DEVICE rechnet alles
auf der CPU (Fine-Tuning dann ein Mehrfaches länger). Die Labs selbst rechnen immer auf der CPU.

Stufen:
    e5      e5-small-Embeddings aller Paragraphen        -> reference/lab4_e5_small.npy (float16)
    main    Fine-Tuning distilbert-german, Seeds 1, 2, 3, drei Epochen, fester Split
            -> reference/lab4_ft.json (Metriken je Epoche, Laufzeiten), lab4_ft_proba.npz
    kopf    nur der Kopf (Backbone eingefroren), Seeds 1, 2, 3 -> Einträge in denselben Dateien
    kopf_lr Kontrolle zu kopf: logistische Regression auf denselben eingefrorenen [CLS]-Zuständen (CPU)
    klein   800 Trainingsparagraphen, Seeds 1, 2, 3: Fine-Tuning im verkürzten Rezept der Übung (2 Epochen,
            Plan über 2 Epochen, 128 Tokens) und im 8-Epochen-Plan mit Validierung (256 Tokens, Epoche auf der
            Validierung gewählt), dazu TF-IDF + LR, auf der Validierung eingestellt
    klein_hebel  800 Paragraphen, Seeds 1, 2, 3, je 2 Epochen (100 Schritte): Token-Variante (256 Tokens, Plan über
            2 Epochen) und Plan-Variante (128 Tokens, Plan über 8 Epochen, nach Epoche 2 gestoppt)
    klein_cv  dieselben 800er-Rezepte auf den fünf Gesetzes-Folds (braucht klein)
    kurz    Fine-Tuning wie main bis zur gewählten Epoche neu gerechnet (CPU), Vorhersagen auf dem Test
            mit vollem und gekürztem Text (erste 100, 200, 400 Zeichen)
    tfidf   TF-IDF + LR: Suche auf der Validierung, dann die Wahl auf den fünf Gesetzes-Folds
    cv      Fine-Tuning Seed 1 auf den fünf Gesetzes-Folds (jedes Gesetz einmal im Test)
    zwillinge  Paare gleichlautender Normen wie in Lab 1 (exakte Dubletten, Jaccard ab 0,75) -> reference/lab4_zwillinge.json
    inferenz  Referenzmessung Inferenzzeit und Modellgröße (CPU, auf einem unbelasteten Rechner ausführen)
Ergebnisse werden zusammengeführt: Eine Stufe überschreibt nur ihre eigenen Einträge.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
import torch

from common import lab4_data as d
from common import lab4_ft as ft

EPOCHS, LR, MAXLEN, BATCH = 3, 5e-5, 256, 16
SMALL_N, SHORT_EPOCHS, SHORT_MAXLEN = 800, 2, 128  # verkürztes Rezept der Übung (live)
SMALL_EPOCHS_MAX = 8                                 # 8-Epochen-Plan mit Validierung; auch Plan der Plan-Variante
TRUNCATION = (100, 200, 400)
SEEDS = tuple(int(x) for x in os.environ.get("LAB4_SEEDS", "1,2,3").split(","))   # Nachrechnen einzelner Seeds
JSON = d.REFERENCE_DIR / "lab4_ft.json"
NPZ = d.REFERENCE_DIR / "lab4_ft_proba.npz"


def _load():
    info = json.loads(JSON.read_text()) if JSON.exists() else {}
    proba = dict(np.load(NPZ)) if NPZ.exists() else {}
    return info, proba


def _save(info, proba):
    """Schreibt die eigenen Einträge über den aktuellen Stand auf der Platte (Stufen dürfen parallel laufen)."""
    d.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    disk_info, disk_proba = _load()
    info = {**disk_info, **info}
    proba = {**disk_proba, **proba}
    info["umgebung"] = {"platform": platform.platform(), "prozessor": platform.processor(),
                        "torch": torch.__version__, "threads": torch.get_num_threads()}
    info["konfig"] = {"epochen": EPOCHS, "lr": LR, "max_length": MAXLEN, "batch_size": BATCH,
                      "klein_n": SMALL_N, "klein_epochen_max": SMALL_EPOCHS_MAX, "klein_max_length": MAXLEN,
                      "klein_kurz_epochen": SHORT_EPOCHS, "klein_kurz_max_length": SHORT_MAXLEN, "kuerzung": TRUNCATION,
                      "klein_hebel": {"tok": {"max_length": MAXLEN, "plan_epochen": SHORT_EPOCHS},
                                      "plan": {"max_length": SHORT_MAXLEN, "plan_epochen": SMALL_EPOCHS_MAX}}}
    JSON.write_text(json.dumps(info, indent=1, ensure_ascii=False))
    np.savez_compressed(NPZ, **{k: v.astype(np.float16) for k, v in proba.items()})


def stage_e5(df):
    for p in (d.REFERENCE_DIR / "lab4_e5_small.npy", d.CACHE_DIR / "lab4_e5_small.npy"):
        p.unlink(missing_ok=True)
    t = time.time()
    E = d.e5_embeddings(df.text.tolist(), device=ft.device)
    d.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(d.REFERENCE_DIR / "lab4_e5_small.npy", E.astype(np.float16))
    info, proba = _load()
    info["e5_sekunden"] = time.time() - t
    info["e5_device"] = ft.device
    _save(info, proba)


def stage_main(df, classes, s):
    info, proba = _load()
    tr, va, te = (s[k] for k in ("train", "val", "test"))
    texts = df.text.tolist()
    for seed in SEEDS:
        tok, model = ft.load_model(classes, seed)
        history = []

        def after_epoch(ep):
            t = time.time()
            for part, idx in (("val", va), ("test", te)):
                p = ft.predict_proba(model, tok, [texts[i] for i in idx], MAXLEN)
                proba[f"ft_s{seed}_e{ep}_{part}"] = p
                m = d.metrics(df.y.values[idx], p.argmax(1))
                history.append({"epoche": ep, "teil": part, **m})
            print(f"seed {seed} epoche {ep}", history[-2:], f"inferenz {time.time() - t:.0f} s", flush=True)

        run_df = ft.train(tok, model, [texts[i] for i in tr], df.y.values[tr], epochs=EPOCHS, lr=LR,
                        batch_size=BATCH, max_length=MAXLEN, seed=seed, on_epoch=after_epoch, log_every=100)
        info[f"ft_s{seed}"] = {"verlauf": history, "sekunden_training": run_df["sekunden"],
                               "schritte": run_df["schritte"], "device": ft.device, "loss_letzte_100": float(np.mean(run_df["losses"][-100:]))}
        _save(info, proba)


def stage_head(df, classes, s):
    """Backbone eingefroren: [CLS]-Zustände einmal berechnen, dann nur pre_classifier und classifier trainieren."""
    info, proba = _load()
    texts = df.text.tolist()
    tok, model = ft.load_model(classes, 1)
    t = time.time()
    H = np.zeros((len(texts), model.config.dim), dtype=np.float32)
    order = np.argsort([len(x) for x in texts])
    with torch.no_grad():
        model.eval()
        for i in range(0, len(texts), 64):
            idx = order[i:i + 64]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(ft.device)
            H[idx] = model.distilbert(**enc).last_hidden_state[:, 0].float().cpu().numpy()
    info["kopf_merkmale_sekunden"] = time.time() - t
    info["kopf_merkmale_device"] = ft.device
    H = torch.tensor(H)
    y = torch.tensor(df.y.values)
    for seed in SEEDS:
        _, model = ft.load_model(classes, seed)
        model = model.to("cpu")                       # der Kopf trainiert auf den Merkmalen im Arbeitsspeicher
        head = torch.nn.ModuleDict({"pre": model.pre_classifier, "cls": model.classifier})
        drop = torch.nn.Dropout(model.config.seq_classif_dropout)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        g = torch.Generator().manual_seed(seed)
        history, top = [], (-1.0, None)

        def logits(idx):
            return head["cls"](drop(torch.relu(head["pre"](H[idx]))))

        for ep in range(1, 31):
            head.train(); drop.train()
            perm = torch.tensor(s["train"])[torch.randperm(len(s["train"]), generator=g)]
            for i in range(0, len(perm), 64):
                idx = perm[i:i + 64]
                loss = torch.nn.functional.cross_entropy(logits(idx), y[idx])
                loss.backward(); opt.step(); opt.zero_grad()
            head.eval(); drop.eval()
            with torch.no_grad():
                pv = logits(torch.tensor(s["val"])).softmax(-1).numpy()
                pt = logits(torch.tensor(s["test"])).softmax(-1).numpy()
            mv = d.metrics(df.y.values[s["val"]], pv.argmax(1))
            history.append({"epoche": ep, "teil": "val", **mv})
            history.append({"epoche": ep, "teil": "test", **d.metrics(df.y.values[s["test"]], pt.argmax(1))})
            if mv["macro_f1"] > top[0]:
                top = (mv["macro_f1"], ep)
                proba[f"kopf_s{seed}_val"], proba[f"kopf_s{seed}_test"] = pv, pt
        n_head = sum(p.numel() for p in head.parameters())
        info[f"kopf_s{seed}"] = {"verlauf": history, "beste_epoche": top[1], "parameter_kopf": n_head}
        print("kopf", seed, top, flush=True)
    _save(info, proba)


def tfidf_on_validation(X_tr, y_tr, X_va, y_va):
    """Dieselbe Suche wie in stage_tfidf (Analyzer, C, Klassengewichte), gewählt nach Macro-F1 auf der Validierung.
    Steht in lab4_data, weil die Übung sie aufruft; Ergebnisse gleich der früheren Fassung hier (geprüft 2026-10-29)."""
    return d.tfidf_tuned(X_tr, y_tr, X_va, y_va)


def stage_small(df, classes, s):
    """800 Paragraphen: verkürztes Rezept (wie live in der Übung) und 8-Epochen-Plan mit Validierung, dazu TF-IDF."""
    info, proba = _load()
    texts, y = df.text.tolist(), df.y.values
    X = df.text.values
    va, te = s["val"], s["test"]
    for seed in SEEDS:
        sub = d.stratified_subset(df, s["train"], SMALL_N, seed=seed)
        # verkürztes Rezept, wie die Teilnehmenden es live rechnen
        tok, model = ft.load_model(classes, seed)
        run_df = ft.train(tok, model, [texts[i] for i in sub], y[sub], epochs=SHORT_EPOCHS, lr=LR,
                        batch_size=BATCH, max_length=SHORT_MAXLEN, seed=seed)
        p = ft.predict_proba(model, tok, [texts[i] for i in te], SHORT_MAXLEN)
        proba[f"klein_kurz_s{seed}_test"] = p
        info[f"klein_kurz_s{seed}"] = {"n": int(len(sub)), "epochen": SHORT_EPOCHS, "max_length": SHORT_MAXLEN,
                                       "schritte": run_df["schritte"], "device": ft.device,
                                       "sekunden_training": run_df["sekunden"], **d.metrics(y[te], p.argmax(1))}
        print("klein kurz", seed, info[f"klein_kurz_s{seed}"], flush=True)
        # 8-Epochen-Plan mit Validierung: 256 Tokens, nach jeder Epoche Validierung, beste Epoche nach Macro-F1
        tok, model = ft.load_model(classes, seed)
        history, top = [], {"macro_f1": -1.0}
        t_eval = [0.0]

        def after_epoch(ep):
            t = time.time()
            for part, idx in (("val", va), ("test", te)):
                pp = ft.predict_proba(model, tok, [texts[i] for i in idx], MAXLEN)
                history.append({"epoche": ep, "teil": part, **d.metrics(y[idx], pp.argmax(1))})
                if part == "val" and history[-1]["macro_f1"] > top["macro_f1"]:
                    top.update(macro_f1=history[-1]["macro_f1"], epoche=ep)
                if part == "test" and top.get("epoche") == ep:
                    proba[f"klein_s{seed}_test"] = pp
            t_eval[0] += time.time() - t
            print(f"klein seed {seed} epoche {ep}", history[-2:], f"{time.time() - t:.0f} s", flush=True)

        run_df = ft.train(tok, model, [texts[i] for i in sub], y[sub], epochs=SMALL_EPOCHS_MAX, lr=LR,
                        batch_size=BATCH, max_length=MAXLEN, seed=seed, on_epoch=after_epoch)
        ep = top["epoche"]
        line = history[2 * ep - 1]
        assert line["teil"] == "test" and line["epoche"] == ep
        info[f"klein_s{seed}"] = {"n": int(len(sub)), "epochen_max": SMALL_EPOCHS_MAX, "max_length": MAXLEN,
                                  "beste_epoche": ep, "verlauf": history, "device": ft.device,
                                  "schritte": run_df["schritte"], "sekunden_training": run_df["sekunden"] - t_eval[0],
                                  "sekunden_bewertung": t_eval[0],
                                  "accuracy": line["accuracy"], "macro_f1": line["macro_f1"]}
        # TF-IDF + LR auf denselben 800, auf der Validierung eingestellt
        t = time.time()
        b, param_grid = tfidf_on_validation(X[sub], y[sub], X[va], y[va])
        proba[f"klein_tfidf_s{seed}_test"] = b["modell"].predict_proba(X[te])
        info[f"klein_tfidf_s{seed}"] = {"analyzer": b["analyzer"], "C": b["C"], "class_weight": b["class_weight"],
                                        "val_macro_f1": b["val_macro_f1"], "gitter": param_grid, "sekunden": time.time() - t,
                                        **d.metrics(y[te], b["modell"].predict(X[te]))}
        print("klein", seed, {k: v for k, v in info[f"klein_s{seed}"].items() if k != "verlauf"},
              "| tfidf", {k: v for k, v in info[f"klein_tfidf_s{seed}"].items() if k != "gitter"}, flush=True)
        _save(info, proba)


class _Stop(Exception):
    pass


def stage_small_levers(df, classes, s):
    """Kontrolle zum Abstand verkürzt gegen 8-Epochen-Plan bei gleicher Schrittzahl (2 Epochen, 100 Schritte):
    tok = nur die Tokenlänge erhöht (256, Plan über 2 Epochen), plan = nur der Plan verlängert (128 Tokens,
    Warmup und Abfall über 8 Epochen ausgelegt, nach Epoche 2 gestoppt). Seeds wie SEEDS, Bewertung auf dem Test."""
    info, proba = _load()
    texts, y = df.text.tolist(), df.y.values
    te = s["test"]
    variants = {"tok": (MAXLEN, SHORT_EPOCHS), "plan": (SHORT_MAXLEN, SMALL_EPOCHS_MAX)}
    for seed in SEEDS:
        sub = d.stratified_subset(df, s["train"], SMALL_N, seed=seed)
        for name, (maxlen, plan) in variants.items():
            tok, model = ft.load_model(classes, seed)
            run_df = ft.train(tok, model, [texts[i] for i in sub], y[sub], epochs=SHORT_EPOCHS, lr=LR,
                            batch_size=BATCH, max_length=maxlen, seed=seed, plan_epochs=plan)
            t = time.time()
            p = ft.predict_proba(model, tok, [texts[i] for i in te], maxlen)
            proba[f"klein_{name}_s{seed}_test"] = p
            info[f"klein_{name}_s{seed}"] = {"n": int(len(sub)), "epochen": SHORT_EPOCHS, "plan_epochen": plan,
                                             "max_length": maxlen, "schritte": run_df["schritte"], "device": ft.device,
                                             "sekunden_training": run_df["sekunden"], "sekunden_test": time.time() - t,
                                             **d.metrics(y[te], p.argmax(1))}
            print(f"klein_{name}", seed, info[f"klein_{name}_s{seed}"], flush=True)
            _save(info, proba)


def stage_short(df, classes, s):
    """Gekürzte Texte: Fine-Tuning wie main bis zur dort auf der Validierung gewählten Epoche neu rechnen
    (die Gewichte von main sind nicht gespeichert) und denselben Lauf auf vollem und gekürztem Test vorhersagen."""
    info, proba = _load()
    texts, y = df.text.tolist(), df.y.values
    tr, te = s["train"], s["test"]
    for seed in SEEDS:
        ep_choice = max((v for v in info[f"ft_s{seed}"]["verlauf"] if v["teil"] == "val"), key=lambda v: v["macro_f1"])["epoche"]
        tok, model = ft.load_model(classes, seed)
        res = {}

        def after_epoch(ep):
            if ep < ep_choice:
                return
            for n in (None, *TRUNCATION):
                txt = [texts[i] if n is None else d.truncate(texts[i], n) for i in te]
                p = ft.predict_proba(model, tok, txt, MAXLEN)
                proba[f"kurz_s{seed}_{n or 'voll'}_test"] = p
                res[n or "voll"] = d.metrics(y[te], p.argmax(1))
            raise _Stop

        t = time.time()
        try:
            ft.train(tok, model, [texts[i] for i in tr], y[tr], epochs=EPOCHS, lr=LR, batch_size=BATCH,
                     max_length=MAXLEN, seed=seed, on_epoch=after_epoch, log_every=100)
        except _Stop:
            pass
        info[f"kurz_s{seed}"] = {"epoche": ep_choice, "device": ft.device, "sekunden": time.time() - t,
                                 "metriken": {str(k): v for k, v in res.items()}}
        print("kurz", seed, info[f"kurz_s{seed}"], flush=True)
        _save(info, proba)


def stage_small_cv(df, classes):
    """800 Paragraphen auf den fünf Gesetzes-Folds, Seeds wie SEEDS: Fine-Tuning im 8-Epochen-Plan (256 Tokens) bis zur
    Epoche, die auf dem festen Split im Median gewählt wurde (Plan über SMALL_EPOCHS_MAX Epochen), und TF-IDF + LR
    in der Einstellung, die auf dem festen Split für denselben Seed gewählt wurde."""
    info, proba = _load()
    texts, y = df.text.tolist(), df.y.values
    X = df.text.values
    ep = int(np.median([info[f"klein_s{sd}"]["beste_epoche"] for sd in (1, 2, 3)]))
    for seed in SEEDS:
        b = info[f"klein_tfidf_s{seed}"]
        t0 = time.time()
        for f, te in enumerate(d.group_folds(df)):
            tr = np.setdiff1d(np.arange(len(df)), te)
            sub = d.stratified_subset(df, tr, SMALL_N, seed=seed)
            tok, model = ft.load_model(classes, seed)

            def after_epoch(e):
                if e == ep:
                    proba[f"klein_cv_s{seed}_f{f}_test"] = ft.predict_proba(model, tok, [texts[i] for i in te], MAXLEN)
                    raise _Stop

            try:
                ft.train(tok, model, [texts[i] for i in sub], y[sub], epochs=SMALL_EPOCHS_MAX, lr=LR,
                         batch_size=BATCH, max_length=MAXLEN, seed=seed, on_epoch=after_epoch)
            except _Stop:
                pass
            m = tfidf(b["analyzer"], b["C"], None if b["class_weight"] == "keine" else b["class_weight"]).fit(X[sub], y[sub])
            proba[f"klein_cv_tfidf_s{seed}_f{f}_test"] = m.predict_proba(X[te])
            info[f"klein_cv_s{seed}_f{f}"] = {
                "epoche": ep, "device": ft.device,
                "ft": d.metrics(y[te], proba[f"klein_cv_s{seed}_f{f}_test"].argmax(1)),
                "tfidf": d.metrics(y[te], proba[f"klein_cv_tfidf_s{seed}_f{f}_test"].argmax(1))}
            print("klein_cv", seed, f, info[f"klein_cv_s{seed}_f{f}"], f"{time.time() - t0:.0f} s", flush=True)
        info[f"klein_cv_s{seed}_sekunden"] = time.time() - t0
        _save(info, proba)


def stage_head_lr(df, classes, s):
    """Kontrolle zu "nur der Kopf": logistische Regression auf denselben eingefrorenen [CLS]-Zuständen,
    standardisiert, C auf der Validierung gewählt."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    info, proba = _load()
    texts, y = df.text.tolist(), df.y.values
    tr, va, te = (s[k] for k in ("train", "val", "test"))
    tok, model = ft.load_model(classes, 1)
    t = time.time()
    H = np.zeros((len(texts), model.config.dim), dtype=np.float32)
    order = np.argsort([len(x) for x in texts])
    with torch.no_grad():
        model.eval()
        for i in range(0, len(texts), 64):
            idx = order[i:i + 64]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(ft.device)
            H[idx] = model.distilbert(**enc).last_hidden_state[:, 0].float().cpu().numpy()
    sec = time.time() - t
    param_grid = {}
    for C in (0.01, 0.1, 1.0):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(H[tr], y[tr])
        param_grid[C] = (d.metrics(y[va], m.predict(H[va]))["macro_f1"], m)
    C = max(param_grid, key=lambda c: param_grid[c][0])
    m = param_grid[C][1]
    proba["kopf_lr_val"], proba["kopf_lr_test"] = m.predict_proba(H[va]), m.predict_proba(H[te])
    info["kopf_lr"] = {"C": C, "val_macro_f1": {str(c): v[0] for c, v in param_grid.items()}, "device": ft.device,
                       "sekunden_merkmale": sec, **d.metrics(y[te], m.predict(H[te]))}
    print("kopf_lr", info["kopf_lr"], flush=True)
    _save(info, proba)


def stage_twins(df):
    """Paare gleichlautender Normen wie in Lab 1, Aufgabe 4: exakte Dubletten im XML-Abzug (Filterkette bis zur
    exakten Stufe) und Paare verschiedener Normen mit Jaccard ab 0,75 auf Wort-5-Grammen. Zeilen wie in d.load()."""
    from common.lab1_kette import clean, five_grams, pairs_above
    from common.lab1_rohkorpus import build_raw_corpus

    raw = build_raw_corpus()
    rest, manifest, _ = clean(raw, until="exakt")
    xml_ids = raw.loc[raw["quelle"] == "gii-xml", "doc_id"].iloc[:len(df)]
    line = {doc: i for i, doc in enumerate(xml_ids)}
    raw_text = raw.set_index("doc_id").text
    assert all(df.text.iloc[i] == raw_text[doc] for doc, i in line.items())
    pairs = []
    for doc, z in manifest.items():
        dup = z["verarbeitung"].get("dublette") or {}
        if z["quelle"] == "gii-xml" and dup.get("art") == "exakt":
            pairs.append(sorted((line[doc], line[dup["von"]])) + [1.0, "exakt"])
    xml = rest[rest.quelle == "gii-xml"]
    grams = {doc: g for doc, g in zip(xml.doc_id, xml.text.map(five_grams)) if g}
    for (a, b), j in pairs_above(grams, 0.75).items():
        i, k = sorted((line[a], line[b]))
        if (df.gesetz[i], df.enbez[i]) != (df.gesetz[k], df.enbez[k]):
            pairs.append([i, k, round(j, 4), "jaccard"])
    pairs.sort()
    (d.REFERENCE_DIR / "lab4_zwillinge.json").write_text(json.dumps(
        {"beschreibung": "Paare gleichlautender Normen (Zeilen von lab4_data.load()): exakte Dubletten im XML-Abzug "
                         "und verschiedene Normen mit Jaccard ab 0,75 auf Wort-5-Grammen, wie in Lab 1",
         "paare": pairs}, ensure_ascii=False))
    print("zwillinge", len(pairs), "Paare", flush=True)


def stage_inference(df, classes, s):
    """Referenzmessung (CPU, ohne Last ausführen): Inferenzzeit je Paragraph und Modellgröße beider Kandidaten."""
    import joblib

    info, proba = _load()
    b = info["tfidf_beste"]
    X, y = df.text.values, df.y.values
    tf = tfidf(b["analyzer"], b["C"], None if b["class_weight"] == "keine" else b["class_weight"]).fit(X[s["train"]], y[s["train"]])
    probe = list(X[s["test"][:256]])
    t = time.time(); tf.predict(probe); sec_tf = (time.time() - t) / len(probe)
    tok, model = ft.load_model(classes, 1)
    ft.predict_proba(model, tok, probe[:32], MAXLEN, batch_size=32)          # Aufwärmen
    t = time.time(); ft.predict_proba(model, tok, probe, MAXLEN, batch_size=32); sec_ft = (time.time() - t) / len(probe)
    d.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tf, d.CACHE_DIR / "tfidf_lr_referenz.joblib")
    info["inferenz_referenz"] = {
        "tfidf_ms": 1000 * sec_tf, "distilbert_ms": 1000 * sec_ft, "faktor": sec_ft / sec_tf,
        "tfidf_mb": d.file_size_mb(d.CACHE_DIR / "tfidf_lr_referenz.joblib"),
        "distilbert_mb": d.file_size_mb(d.DBERT / "model.safetensors"),
        "device": ft.device, "threads": torch.get_num_threads(), "platform": platform.platform(),
        "prozessor": platform.processor() or platform.machine()}
    print(info["inferenz_referenz"], flush=True)
    _save(info, proba)


def stage_cv(df, classes):
    info, proba = _load()
    texts = df.text.tolist()
    ep_fixed = info.get("cv_epochen", EPOCHS)
    for f, te in enumerate(d.group_folds(df)):
        tr = np.setdiff1d(np.arange(len(df)), te)
        tok, model = ft.load_model(classes, 1)
        run_df = ft.train(tok, model, [texts[i] for i in tr], df.y.values[tr], epochs=ep_fixed, lr=LR,
                        batch_size=BATCH, max_length=MAXLEN, seed=1, log_every=200)
        p = ft.predict_proba(model, tok, [texts[i] for i in te], MAXLEN)
        proba[f"cv_f{f}_test"] = p
        info[f"cv_f{f}"] = {"epochen": ep_fixed, "device": ft.device, "sekunden_training": run_df["sekunden"],
                            **d.metrics(df.y.values[te], p.argmax(1))}
        print("cv", f, info[f"cv_f{f}"], flush=True)
        _save(info, proba)


def tfidf(analyzer="char_wb", C=1.0, class_weight=None):
    """TF-IDF + logistische Regression wie auf der Folie; analyzer, C und class_weight wählbar."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(d.tfidf_vectorizer(analyzer), LogisticRegression(C=C, class_weight=class_weight, max_iter=3000))


def stage_tfidf(df, s):
    """Suche über Analyzer, C und Klassengewichte auf der Validierung; danach Gesetzes-Folds mit der Wahl."""
    X, y = df.text.values, df.y.values
    tr, va, te = (s[k] for k in ("train", "val", "test"))
    param_grid = []
    for analyzer in ("char_wb", "word"):
        for cw in (None, "balanced"):
            for C in (1.0, 10.0, 100.0):
                t = time.time()
                m = tfidf(analyzer, C, cw).fit(X[tr], y[tr])
                param_grid.append({"analyzer": analyzer, "C": C, "class_weight": cw or "keine",
                               "val": d.metrics(y[va], m.predict(X[va])),
                               "test": d.metrics(y[te], m.predict(X[te])), "sekunden": time.time() - t})
                print(param_grid[-1], flush=True)
    top = max(param_grid, key=lambda g: g["val"]["macro_f1"])
    info, proba = _load()
    info["tfidf_gitter"], info["tfidf_beste"] = param_grid, top
    cw = None if top["class_weight"] == "keine" else top["class_weight"]
    t = time.time()
    m = tfidf(top["analyzer"], top["C"], cw).fit(X[tr], y[tr])
    info["tfidf_beste"]["sekunden_fit"] = time.time() - t
    proba["tfidf_val"], proba["tfidf_test"] = m.predict_proba(X[va]), m.predict_proba(X[te])
    for f, te_f in enumerate(d.group_folds(df)):
        tr_f = np.setdiff1d(np.arange(len(df)), te_f)
        m = tfidf(top["analyzer"], top["C"], cw).fit(X[tr_f], y[tr_f])
        proba[f"cv_tfidf_f{f}_test"] = m.predict_proba(X[te_f])
        info[f"cv_tfidf_f{f}"] = d.metrics(y[te_f], m.predict(X[te_f]))
        print("cv tfidf", f, info[f"cv_tfidf_f{f}"], flush=True)
    _save(info, proba)


def main(stages):
    df, classes = d.load()
    s = d.split(df)
    if "e5" in stages:
        stage_e5(df)
    if "main" in stages:
        stage_main(df, classes, s)
    if "kopf" in stages:
        stage_head(df, classes, s)
    if "kopf_lr" in stages:
        stage_head_lr(df, classes, s)
    if "klein" in stages:
        stage_small(df, classes, s)
    if "kurz" in stages:
        stage_short(df, classes, s)
    if "klein_hebel" in stages:
        stage_small_levers(df, classes, s)
    if "klein_cv" in stages:
        stage_small_cv(df, classes)
    if "zwillinge" in stages:
        stage_twins(df)
    if "tfidf" in stages:
        stage_tfidf(df, s)
    if "cv" in stages:
        stage_cv(df, classes)
    if "inferenz" in stages:
        stage_inference(df, classes, s)


if __name__ == "__main__":
    main(sys.argv[1:] or ["e5", "tfidf", "main", "kopf", "kopf_lr", "klein", "klein_hebel", "kurz", "cv", "klein_cv", "zwillinge", "inferenz"])
