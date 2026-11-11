
assert tok.get_vocab_size() == 8000, "Lab 3 erwartet V = 8.000: Abschnitt 2 mit vocab_size=8000 ausführen"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
tok.save(str(l2.TOKENIZER_JSON))
eos = tok.token_to_id("</s>")
dtype = np.uint16 if tok.get_vocab_size() <= 2**16 else np.uint32
for texts, path in [(train, l2.TRAIN_BIN), (holdout, l2.VAL_BIN)]:
    ids = []
    for e in tok.encode_batch(texts):
        ids += e.ids + [eos]
    np.array(ids, dtype=dtype).tofile(path)
    print(f"{path.name:16s} {format_de(len(ids)):>10} Tokens  {format_de(path.stat().st_size / 1e6, 1):>5} MB")
print(f"{l2.TOKENIZER_JSON.name:16s} {format_de(l2.TOKENIZER_JSON.stat().st_size / 1e6, 1):>16} MB")
