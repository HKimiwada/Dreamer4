"""
High-level overview:
    Given (patch_tokens, mask), how does the model encode visible patches, 
    compress them temporally, and then reconstruct the missing ones?
encoder_decoder.py:
    Defines how the tokenizer as a whole works (high-level pipeline). 
    Uses modules defined in transformer_blocks.py. 
Conceptual Overview:
    1. Input (masked patches)
    2. Stack of spatial + temporal blocks (encoder)
    3. tanh bottleneck
    4. Stack of spatial + temporal blocks (decoder)
    5. Linear projection 
    6. Output (reconstructed patches)
"""
