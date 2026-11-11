sentence = "Der Käufer ist verpflichtet, dem Verkäufer den vereinbarten Kaufpreis zu zahlen."
enc = tok.encode(sentence)
print(len(enc.ids), "Tokens:", enc.tokens)
print("IDs:", enc.ids)
print("decode gleich Eingabe:", tok.decode(enc.ids) == sentence)




gpt2 = AutoTokenizer.from_pretrained(MODELS_DIR / "gpt2")
gbert = AutoTokenizer.from_pretrained(MODELS_DIR / "gbert-base", tokenizer_type="bert")
for name, t in [("eigener BPE", tok), ("gpt2", gpt2), ("gbert-base", gbert)]:
    size = t.get_vocab_size() if hasattr(t, "get_vocab_size") else len(t)
    print(f"{name:12s} {format_de(size):>7} Einträge")



n_chars = sum(map(len, holdout))
words = sum(len(t.split()) for t in holdout)
count = {
    "eigener BPE": sum(len(e.ids) for e in tok.encode_batch(holdout)),
    "gpt2": sum(map(len, gpt2(holdout, add_special_tokens=False)["input_ids"])),
    "gbert-base": sum(map(len, gbert(holdout, add_special_tokens=False)["input_ids"])),
}
print(f"Holdout: {format_de(n_chars)} Zeichen, {format_de(words)} Wörter\n")
for name, n in count.items():
    print(f"{name:12s} {format_de(n):>9} Tokens   {format_de(n_chars / n)} Zeichen je Token   {format_de(n / words)} Tokens je Wort")
nl = "\n"
per_break = {name: len(l2.ids_of(t, [nl])[0]) for name, t in [("eigener BPE", tok), ("gpt2", gpt2), ("gbert-base", gbert)]}
print(f"\n{format_de(sum(t.count(nl) for t in holdout))} Zeilenumbrüche im Holdout, Tokens je Umbruch:", per_break)




compound_words = l2.compounds(train, holdout, with_split=True)
print(f"{len(compound_words)} Komposita im Holdout, zum Beispiel:")
random.seed(SEED)
for word, (left, linker, right_part) in random.sample(compound_words, 10):
    print(f"  {word:28s} = {left} + {linker + ' + ' if linker else ''}{right_part}")
words_comp = [w for w, _ in compound_words]



for word in ["Kündigungsschutzgesetz", "Grundstücksverkehrsgesetz", "Verbraucherdarlehensvertrag"]:
    print(word)
    print("  eigener BPE ", tok.encode(" " + word).tokens)
    print("  gpt2        ", gpt2.tokenize(" " + word))
    print("  gbert-base  ", gbert.tokenize(word))
print()
for name, t in [("eigener BPE", tok), ("gpt2", gpt2), ("gbert-base", gbert)]:
    ids = l2.ids_of(t, [" " + w for w in words_comp])
    share = sum(len(i) == 1 for i in ids) / len(ids)
    print(f"{name:12s} Komposita in einem Token {format_de(100 * share, 1):>5} %"
          f"   Tokens je Kompositum {format_de(np.mean([len(i) for i in ids]))}")
