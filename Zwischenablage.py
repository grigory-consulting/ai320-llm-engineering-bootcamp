sentence = "Der Käufer ist verpflichtet, dem Verkäufer den vereinbarten Kaufpreis zu zahlen."
enc = tok.encode(sentence)
print(len(enc.ids), "Tokens:", enc.tokens)
print("IDs:", enc.ids)
print("decode gleich Eingabe:", tok.decode(enc.ids) == sentence)
