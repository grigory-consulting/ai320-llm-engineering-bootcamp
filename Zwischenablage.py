import json

ref = Path("reference")
prep = json.loads((ref / "lab3_langer_lauf.json").read_text(encoding="utf-8"))
sd = torch.load(ref / "lab3_langer_lauf.pt", weights_only=True)    # nur Tensoren: weights_only=True genügt
sd = {k: v.float() for k, v in sd.items()}
sd["head.weight"] = sd["tok.weight"]                                # Weight Tying: eine Matrix, zwei Schlüssel
model_p = MiniGPT().to(device)
model_p.load_state_dict(sd)
model_p.eval()

# Tokenizer und Holdout passend zum vorbereiteten Lauf (gleicher Split, gleicher Tokenizer)
tok_p, val_p = lab3_data.prepared_tokenizer_and_holdout(prep, data)
print(prep["config"])



@torch.no_grad()
def generate(model, prompt, n_new=80, temperature=0.8):
    idx = torch.tensor([tok_p.encode(prompt).ids], device=device)
    for _ in range(n_new):
        logits = model(idx[:, -256:])[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
    return tok_p.decode(idx[0].tolist())


torch.manual_seed(SEED)
for temperature in (0.8, 1.5):
    print(f"--- Temperatur {temperature}")
    print(generate(model_p, "§ 12 Pflichten des Arbeitgebers\n(1) Der Arbeitgeber", temperature=temperature))
