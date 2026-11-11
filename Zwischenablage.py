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
