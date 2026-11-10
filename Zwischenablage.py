




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


