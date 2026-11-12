





log = []
t0 = time.perf_counter()
for step in range(0, 100):
    row = train_step(model, optimizer, scheduler, step)
    log.append(row)
    if (step + 1) % 20 == 0:
        print(f"Schritt {step + 1:4d}  Loss {row['loss']:.3f}  Grad-Norm {row['grad_norm']:.2f}  LR {row['lr']:.2e}")
print(f"{time.perf_counter() - t0:.1f} s")


def plot_log(log, title=""):
    fig, ax = plt.subplots(3, 1, figsize=(8, 5.5), sharex=True)
    for a, key, label in zip(ax, ["loss", "grad_norm", "lr"], ["Loss", "Grad-Norm", "Lernrate"]):
        a.plot([r[key] for r in log], lw=1)
        a.set_ylabel(label)
    ax[1].axhline(config["clip"], color="grey", ls="--", lw=0.8)
    ax[-1].set_xlabel("Schritt")
    ax[0].set_title(title)
    plt.tight_layout()
    plt.show()


plot_log(log, "Erste 100 Schritte")
