@torch.no_grad()
def holdout_perplexity(model, val, seq_len=256):
    n = (len(val) - 1) // seq_len
    w = torch.from_numpy(val[: n * seq_len + 1].astype(np.int64))
    x, y = w[:-1].view(n, seq_len), w[1:].view(n, seq_len)
    total = sum(F.cross_entropy(model(xb).flatten(0, 1), yb.flatten(), reduction="sum").item()
                for xb, yb in zip(x.split(16), y.split(16)))
    loss = total / y.numel()
    return loss, math.exp(loss), y.numel()


t0 = time.perf_counter()
loss_p, ppl_p, n_tok = holdout_perplexity(model_p, val_p)
print(f"Holdout: {lab3_data.format_de(n_tok)} Zielpositionen, Loss {loss_p:.3f}, Perplexity {ppl_p:.1f}  ({time.perf_counter() - t0:.1f} s)")
