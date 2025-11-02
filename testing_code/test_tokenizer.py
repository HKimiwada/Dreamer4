# python testing_code/test_tokenizer.py
import torch
from tokenizer.model.encoder_decoder import CausalTokenizer

model = CausalTokenizer(input_dim=768, embed_dim=768, num_heads=8, num_layers=12, latent_dim=256)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
print(model.encoder[0], "\n\n", model.decoder[-1])

print("\nTesting forward pass...")
model = CausalTokenizer()
patches = torch.randn(2, 4, 8, 768)   # B=2, T=4, N=8, D_in=768
mask = torch.rand(2, 4, 8) > 0.5
y = model(patches, mask)
print(y.shape)  # should be (2, 4, 8, 768)
