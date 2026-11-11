FUNCTION_WORDS = set("der die das und ist nicht mit von den dem des sich wird werden oder auf für eine einer "
                       "einem zu nach bei durch als auch wenn zur zum im ein sind bis nur wie über aus kann hat".split())
LINK = re.compile(r"https?://\S+")


def language_score(text: str) -> float:
    words = re.findall(r"[a-zäöüß]+", text.lower())
    return sum(w in FUNCTION_WORDS for w in words) / max(1, len(words))


def duplicate_line_share(text: str) -> float:              # FineWeb: Zeichenanteil mehrfach vorkommender Zeilen
    rows = [z.strip() for z in text.split("\n") if z.strip()]
    duplicate = {z for z in rows if rows.count(z) > 1}
    return sum(len(z) for z in rows if z in duplicate) / max(1, sum(len(z) for z in rows))


score = df["text"].map(language_score)
for d, s in zip(df["doc_id"], score):
    log[d]["sprache"] = round(s, 3)
ads = df["text"].map(lambda t: duplicate_line_share(t) >= 0.1 or len(LINK.findall(t)) >= 2)
foreign = ~ads & (df["text"].str.split().str.len() >= 10) & (score < 0.1)
result = pd.Series("behalten", index=df.index).mask(ads, "werbung").mask(foreign, "sprache").rename("Ergebnis")
print(pd.crosstab(df["quelle"], result))
print("\nSprachscore der Normen: Median", round(score[df["quelle"] == "gii-xml"].median(), 2))
forum = df["quelle"] == "web-forum"
print("\n--- Forum, erste Zeile je Beitrag ---")
print(pd.DataFrame({"Ergebnis": result[forum], "text": df.loc[forum, "text"].str.split("\n").str[0].str[:75]})
      .to_string(index=False))
df = drop(df, {d: r for d, r in zip(df["doc_id"], result) if r != "behalten"})



EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
IBAN = re.compile(r"\bDE\d{2}(?: ?\d{4}){4} ?\d{2}\b")


def mask(text: str) -> tuple[str, dict]:
    n_mail = n_iban = 0
    if "@" in text:
        text, n_mail = EMAIL.subn("<EMAIL>", text)
    if "DE" in text:
        text, n_iban = IBAN.subn("<IBAN>", text)
    return text, {"EMAIL": n_mail, "IBAN": n_iban}


fresh = df["text"].map(mask)                              # je Dokument: (maskierter Text, Zählung)
df["text"] = fresh.str[0]
for d, n in zip(df["doc_id"], fresh.str[1]):
    log[d]["pii"] = n
print(pd.DataFrame(fresh.str[1].tolist()).assign(quelle=df["quelle"]).groupby("quelle").sum())
print("\n" + df.loc[df["quelle"] == "intern-synthetisch", "text"].iloc[30])






def dedup_key(text: str) -> str:
    norm = unicodedata.normalize("NFC", text).lower()
    norm = " ".join(norm.split())                          # Leerraum vereinheitlichen
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


raw_hash = df["text"].map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
df["sha256"] = df["text"].map(dedup_key)
duplicate = df["sha256"].duplicated()
first = df.drop_duplicates("sha256").set_index("sha256")["doc_id"]   # Hash -> erstes Dokument mit diesem Text
for d, h, dup in zip(df["doc_id"], df["sha256"], duplicate):
    log[d]["sha256"] = h
    if dup:
        log[d]["dublette"] = {"art": "exakt", "von": first[h], "jaccard": 1.0}
print("Dubletten nach Roh-Bytes:       ", raw_hash.duplicated().sum())
print("Dubletten nach Normalisierung:  ", duplicate.sum())
exact_by_source = df.loc[duplicate, "quelle"].value_counts().to_dict()
df = drop(df, dict.fromkeys(df.loc[duplicate, "doc_id"], "dublette-exakt"))
print("verbleiben:", len(df))





from datasketch import MinHash, MinHashLSH


def five_grams(text: str) -> set[str]:
    w = text.split()
    return {" ".join(w[i:i + 5]) for i in range(len(w) - 4)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


lsh = MinHashLSH(num_perm=112, params=(14, 8))
grams, removed = {}, []                                    # grams: 5-Gramme der behaltenen Dokumente
for d, t in zip(df["doc_id"], df["text"]):
    g = five_grams(t)
    if not g:                                              # unter fünf Wörtern: nur der exakte Hash
        continue
    m = MinHash(num_perm=112, seed=1)
    m.update_batch([x.encode("utf-8") for x in g])
    hits = [(jaccard(g, grams[c]), c) for c in lsh.query(m)]   # Kandidaten, exakt nachgemessen
    hits = [x for x in hits if x[0] >= 0.75]
    if hits:
        j, c = max(hits)
        log[d]["dublette"] = {"art": "beinahe", "von": c, "jaccard": round(j, 3)}
        removed.append(d)
    else:
        lsh.insert(d, m)
        grams[d] = g


idx = df.set_index("doc_id")


def pair_kind(d: str) -> str:
    a, b = idx.loc[d], idx.loc[log[d]["dublette"]["von"]]
    if a["quelle"] == "gii-html" and a["norm"] == b["norm"]:
        return "HTML-Kopie derselben Norm"
    return "andere Norm" if a["quelle"].startswith("gii") else a["quelle"]


kind = {d: pair_kind(d) for d in removed}
print(pd.Series(kind).value_counts().to_string())
df = drop(df, dict.fromkeys(removed, "dublette-beinahe"))
original = dict(zip(df.loc[df["quelle"] == "gii-xml", "norm"], df.loc[df["quelle"] == "gii-xml", "text"]))
rest = df[df["quelle"] == "gii-html"]
j_rest = pd.Series([jaccard(five_grams(t), five_grams(original[n]))
                    for n, t in zip(rest["norm"], rest["text"]) if n in original])
print(f"HTML-Kopien noch im Korpus: {len(rest)}, Wörter im Median {rest['text'].str.split().str.len().median():.0f}, "
      f"Jaccard zum Original: Median {j_rest.median():.2f}, davon >= 0,75: {(j_rest >= 0.75).sum()}")
print("verbleiben insgesamt:", len(df))
examples = sorted(((log[d]["dublette"]["jaccard"], d) for d in removed if kind[d] == "andere Norm"), reverse=True)
for j, d in examples:
    print(f"  {j:.2f}  {idx.loc[d, 'norm']}  ~  {idx.loc[log[d]['dublette']['von'], 'norm']}")


j_max, d_max = examples[0]                                 # das ähnlichste Paar aus zwei verschiedenen Normen
for d in (log[d_max]["dublette"]["von"], d_max):
    print(f"{idx.loc[d, 'norm']}  sha256 {idx.loc[d, 'sha256'][:16]}…\n{idx.loc[d, 'text']}\n")
print(f"Jaccard {j_max:.3f}")




