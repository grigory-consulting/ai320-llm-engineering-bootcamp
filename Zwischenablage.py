import os, sys
from pathlib import Path

for candidate in (Path.cwd(), Path.cwd() / "labs", *Path.cwd().parents):
    if (candidate / "common" / "paths.py").exists():
        os.chdir(candidate)
        sys.path.insert(0, str(candidate))
        break

import json, re, time
import numpy as np
import pandas as pd
from common import corpus, lab5_rag as lab5
from common.paths import MODELS_DIR, SEED

device = "cpu"
pd.set_option("display.width", 140, "display.max_colwidth", 90)
print("Lab-Ordner:", Path.cwd())





norms = lab5.index_norms(corpus.load())


def chunks(df, max_chars=1500):
    rows = []
    for r in df.itertuples():
        nid = lab5.norm_id(r.gesetz, r.enbez)                  # "BGB", "§ 438" -> "bgb#438"
        head = f"{r.gesetz} {r.enbez} {r.titel}".strip()
        paragraphs = r.text.split("\n")
        if len(r.text) <= max_chars or len(paragraphs) == 1:
            pieces = [(nid, r.text)]
        else:
            pieces = [(f"{nid}-{i}", a) for i, a in enumerate(paragraphs, 1)]
        for cid, text in pieces:
            rows.append({"id": cid, "norm": nid, "text": f"{head}\n{text}", "builddate": r.builddate})
    return pd.DataFrame(rows)


ch = chunks(norms)
text_of = dict(zip(ch.id, ch.text))
print(f"{len(norms)} Normen -> {len(ch)} Chunks, im Mittel {ch.text.str.len().mean():.0f} Zeichen")
print(ch[ch.norm.isin(["bgb#438", "bgb#622"])][["id", "text"]].assign(text=lambda d: d.text.str[:80]).to_string(index=False))




from sentence_transformers import SentenceTransformer

model = SentenceTransformer(str(MODELS_DIR / "e5-small"), device=device)
E = lab5.embeddings("gemischt", ch,
                    lambda: model.encode(["passage: " + t for t in ch.text], batch_size=32,
                                         normalize_embeddings=True, show_progress_bar=True))
ids = ch.id.tolist()
print("Matrix:", E.shape, E.dtype)


def dense(question, k=50):
    q = model.encode(["query: " + question], normalize_embeddings=True)[0]
    scores = E @ q                                              # Kosinus = Skalarprodukt
    return [ids[i] for i in np.argsort(-scores, kind="stable")[:k]]


question = "Wie lange kann ich Mängel an einer gekauften Sache geltend machen?"
for cid in dense(question, k=5):
    print(f"{cid:12s} {text_of[cid][:90]}")
