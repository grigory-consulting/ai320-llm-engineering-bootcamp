
def get_batch(data, batch, seq_len):
    i = torch.randint(len(data) - seq_len - 1, (batch,))
    w = torch.stack([torch.from_numpy(data[j : j + seq_len + 1].astype(np.int64)) for j in i.tolist()])
    return w[:, :-1].to(device), w[:, 1:].to(device)
