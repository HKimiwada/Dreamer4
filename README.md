# Dreamer4 Implementation
Custom implementation of "Training Agents Inside of Scalable World Models" originally published by Google DeepMind.
Trained on 8x 16GB V100 GPUs using data from zhwang4ai/OpenAI-Minecraft-Contractor.

Key Differences:
1. Uses MSE loss only when training tokenizer (no dynamic LPIPS integration as of yet)
2. Uses simple 33-dim action token. Instead of the multi-token encoding approach used by the paper