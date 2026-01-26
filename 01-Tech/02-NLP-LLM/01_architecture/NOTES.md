# Architecture Notes

Focus: base building blocks of transformer models and common variants.

- Positional encoding: sinusoidal, RoPE, ALiBi, relative position bias.
- Normalization: LayerNorm and RMSNorm stabilize deep nets.
- Attention: scaled dot-product attention with masking, MHA, MQA, GQA.
- Feedforward: MLP with GELU/GEGLU/SwiGLU variants and activation functions.
- MoE: gate tokens to multiple experts with sparse routing.
- Transformer block: pre-norm block with attention + MLP.
- Embedding and weight tying: share token embedding and output projection.
