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
