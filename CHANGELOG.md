# Changelog

All notable changes to CRUX are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2026-01-24

### Added
- Revised config file, updated args parsing in main training script along with it
- Reduced default sequence length to 512
- Recursive patterns received LayerNorm
- Integrate AMP for mixed precision training
- Added gradient checkpointing
- Added model-level configuration params to `config.yaml`
- Revised scheduling and composition of Muon and AdamW
- Weight decay directly gets scaled by the learning rate now
- Revised `train.py` to use `torch.compile` if configured
- Completed maiden training run with `config.yaml` configuration (see [metrics](https://huggingface.co/Marcus2112/crux/blob/main/logs/metrics.jsonl))
- Reset to mask-token usage (for now, stabilizes training unreasonably well)

## [0.0.1] - 2026-01-21

### Added
- Initial release of CRUX
- 10-layer transformer model (384 hidden size, 8 attention heads)
- 16-step diffusion denoising
- Attention sink mechanism
- ColBERT projection head for retrieval tasks
- Training and evaluation scripts
- Visualization tools for diffusion process

### Known Issues
- Training on GPU only (CPU support untested)
- Limited to GPT-2 tokenizer vocabulary (32,000 tokens)
- Checkpoint paths hardcoded in visualization scripts

### Performance Notes
- Model tested with sequence length up to 512
