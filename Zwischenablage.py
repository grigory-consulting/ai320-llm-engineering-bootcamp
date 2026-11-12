
def get_batch(data, batch, seq_len):
    i = torch.randint(len(data) - seq_len - 1, (batch,))
    w = torch.stack([torch.from_numpy(data[j : j + seq_len + 1].astype(np.int64)) for j in i.tolist()])
    return w[:, :-1].to(device), w[:, 1:].to(device)




config = {"seed": SEED, "batch": 16, "seq_len": 256, "lr": 1e-3, "warmup": 100, "total_steps": 2000,
          "min_ratio": 0.1, "weight_decay": 0.1, "betas": (0.9, 0.95), "clip": 1.0, "device": device}


def lr_factor(step):
    if step < config["warmup"]:
        return (step + 1) / (config["warmup"] + 1)
    progress = min(1.0, (step - config["warmup"]) / (config["total_steps"] - config["warmup"]))
    return config["min_ratio"] + 0.5 * (1 - config["min_ratio"]) * (1 + math.cos(math.pi * progress))


def build(config):
    torch.manual_seed(config["seed"])
    model = MiniGPT().to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]       # Matrizen, Embeddings
    no_decay = [p for p in model.parameters() if p.dim() < 2]     # RMSNorm-Gewichte
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=config["lr"], betas=config["betas"], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    return model, optimizer, scheduler


model, optimizer, scheduler = build(config)








def train_step(model, optimizer, scheduler, step):
    x, y = get_batch(train_data, config["batch"], config["seq_len"])
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
    lr = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": loss.item(), "grad_norm": grad_norm.item(), "lr": lr}


@torch.no_grad()
def holdout_loss(model, n=128, seq_len=256):
    model.eval()
    starts = np.linspace(0, len(val_data) - seq_len - 2, n).astype(np.int64)
    w = torch.stack([torch.from_numpy(val_data[j : j + seq_len + 1].astype(np.int64)) for j in starts])
    losses = [F.cross_entropy(model(b[:, :-1]).flatten(0, 1), b[:, 1:].flatten()).item() for b in w.split(16)]
    model.train()
    return float(np.mean(losses))

