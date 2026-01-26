# LLM Knowledge Map

This folder organizes core LLM concepts into small, runnable Python/PyTorch examples.
Each topic has its own `.py` file with a minimal demo in `__main__`.

## 01 Architecture
- `01_architecture/NOTES.md`
- `01_architecture/positional_encoding_sinusoidal.py`
- `01_architecture/positional_encoding_rope.py`
- `01_architecture/positional_encoding_alibi.py`
- `01_architecture/relative_position_bias.py`
- `01_architecture/layer_norm.py`
- `01_architecture/self_attention.py`
- `01_architecture/multi_head_attention.py`
- `01_architecture/mqa_attention.py`
- `01_architecture/gqa_attention.py`
- `01_architecture/feedforward_mlp.py`
- `01_architecture/activations.py`
- `01_architecture/moe_ffn.py`
- `01_architecture/transformer_block.py`
- `01_architecture/embedding_and_tying.py`

## 02 Pretraining
- `02_pretraining/NOTES.md`
- `02_pretraining/data_tokenization.py`
- `02_pretraining/packing_and_masking.py`
- `02_pretraining/causal_lm_loss.py`
- `02_pretraining/optimizer_adamw.py`
- `02_pretraining/lr_schedule.py`
- `02_pretraining/grad_accumulation.py`
- `02_pretraining/mixed_precision.py`
- `02_pretraining/checkpointing.py`

## 03 Acceleration
- `03_acceleration/NOTES.md`
- `03_acceleration/flash_attention_demo.py`
- `03_acceleration/lora_linear.py`
- `03_acceleration/quantization_int8.py`
- `03_acceleration/kv_cache.py`

Run any file directly, for example:

```bash
python 01_architecture/self_attention.py
```
