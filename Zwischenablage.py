




class PatchEmbedding(nn.Module):
    def __init__(self, img_size = 32, patch_size = 4, in_channels = 3, embed_dim = 128 ): # 
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride= patch_size)
    
    def forward(self,x):
        x = self.proj(x) # (B, embed_dim, img_size/patch_size, img_size/patch_size )
        x = x.flatten(2).transpose(1,2) # (B, num_patches, embed_dim)
        return x 




model = VisionTransformer(img_size=32, patch_size=4, num_classes=10, embed_dim=128, depth=6, num_heads=4, drop=0.1)
model.to(torch.device(device))

lr = 3e-4
epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {num_params/1e6:.2f}M")



def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping 
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.inference_mode()
def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total



for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, device)
    scheduler.step()
    print(f"Epoch {epoch + 1}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")





