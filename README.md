# Dreamer4 Implementation
Custom implementation of "Training Agents Inside of Scalable World Models" originally published by Google DeepMind.
Trained on 8x 16GB V100 GPUs using data from zhwang4ai/OpenAI-Minecraft-Contractor.

## Key Differences from original paper:
- Uses MSE loss only when training tokenizer (no dynamic LPIPS integration as of yet)
- Tokenizer/World Model/Imagination Training etc... all overfit on one video to prove the pipeline works on limited compute resources.
- Due to limited computing resources, tokenizer and world model outputs are grainy than ideal reconstructions.

## Results
### Tokenizer Performance
Example #1: Reconstruction from Tokenizer <br>

https://github.com/user-attachments/assets/d66c2a8b-a857-441c-9626-95f70268b5d1

Example #2: Reconstruction from Tokenizer <br>

https://github.com/user-attachments/assets/a3cd04e5-7409-4a6f-b1fb-259e335cf879

### World Model Performance
Single-step inference (shows that overfitting worked)

https://github.com/user-attachments/assets/a0e15de8-39d4-41ea-8426-853545e1a711

Multi-step inference (simulates actual world model performance)

https://github.com/user-attachments/assets/ac0abd4c-5e62-4d06-b088-df7cd9373bb8


