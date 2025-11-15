with open("data/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl") as f:
    num_lines = sum(1 for _ in f)
print("Total lines:", num_lines)

import torch
import json
ckpt = "data/latent_sequences/video_00000.pt"
test = torch.load(ckpt)
print(test)
print()

action_ckpt = "data/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl"
with open(action_ckpt, "r") as f:
    for i, line in enumerate(f):
        if i == 10:
            break
        entry = json.loads(line)
        print(f"Action {i}: {entry}")