# Dreamer4 Implementation
Custom implementation of "Training Agents Inside of Scalable World Models" originally published by Google DeepMind.
Trained on 8x 16GB V100 GPUs using data from zhwang4ai/OpenAI-Minecraft-Contractor.

## Key Differences from original paper:
- Uses MSE loss only when training tokenizer (no dynamic LPIPS integration as of yet)
- Tokenizer/World Model/Imagination Training etc... all overfit on one video to prove the pipeline works on limited compute resources.

## Results
### Tokenizer Performance
Example #1: Reconstruction from Tokenizer <br>

https://github.com/user-attachments/assets/d66c2a8b-a857-441c-9626-95f70268b5d1

Example #2: Reconstruction from Tokenizer <br>

https://github.com/user-attachments/assets/a3cd04e5-7409-4a6f-b1fb-259e335cf879

### World Model Performance
