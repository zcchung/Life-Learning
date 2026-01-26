# Pretraining Notes

Focus: from data to training loops and the smallest reusable pieces.

- Tokenization: toy character-level mapping and dataset prep.
- Packing and masking: pad/causal masks for batching.
- Causal LM loss: shift logits and labels.
- AdamW: parameter grouping for weight decay.
- LR schedule: warmup + cosine decay.
- Gradient accumulation: simulate large batch size.
- Mixed precision: autocast + GradScaler.
- Checkpointing: save/load model and optimizer state.
