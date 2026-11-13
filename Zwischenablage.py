





from rank_bm25 import BM25Okapi


def tokens(text):
    return re.findall(r"\w+", text.lower())


bm25 = BM25Okapi([tokens(t) for t in ch.text])


def bm25_search(question, k=50):
    scores = bm25.get_scores(tokens(question))
    return [ids[i] for i in np.argsort(-scores, kind="stable")[:k]]


def hybrid(question, k=50):
    return lab5.rrf(dense(question, 50), bm25_search(question, 50), k=60)[:k]


def rank(ranking, gold):                     # erster Chunk, der zu einer Gold-Norm gehört
    for i, cid in enumerate(ranking, start=1):
        if lab5.norm_of(cid) in gold:
            return i
    return None




def compose(question, hits):
    sources = "\n\n".join(f'<quelle id="{d}">\n{text_of[d][:2500]}\n</quelle>' for d in hits)
    system = ("Beantworten Sie die Frage nur aus den Quellen. "
              "Belegen Sie jede Aussage mit der id der Quelle in eckigen Klammern. "
              "Steht die Antwort nicht in den Quellen, antworten Sie: Nicht in den Quellen.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{sources}\n\nFrage: {question}"}]


best_row = table_df.MRR.idxmax()
search = retriever[best_row]
print("Retriever für den Kontext:", best_row)



cl, GENERATOR = lab5.llm_client()


def generate(messages):
    return lab5.cached(GENERATOR, messages, lambda: cl.chat.completions.create(
        model=GENERATOR, messages=messages, temperature=0).choices[0].message.content, temperature=0)


question = gold_questions[5]["frage"]
top5 = search(question, k=5)
answer = generate(compose(question, top5))
cited = set(lab5.cited_ids(answer))
print("Generator:", GENERATOR, "\n" + question, "\nKontext:", top5, "\n\n" + answer)
print("\nBelege außerhalb des Kontexts:", cited - set(top5))
