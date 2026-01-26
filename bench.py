import sys
import torch
import argparse

from pathlib import Path
from lm_eval.api.model import LM
from transformers import AutoTokenizer
from lm_eval import utils, simple_evaluate

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import set_seeds
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


class CruxHFLM(LM):
    def __init__(self, model, tokenizer, diffusion_steps=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = 1

        # Match training dtype
        self.dtype = torch.float32
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16

        self.model.to(self.dtype)
        self._rank = 0
        self._world_size = 1
        self.diffusion_steps = diffusion_steps

    @property
    def vocab_size(self):
        return self.model.vocab_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 512

    def loglikelihood(self, requests):
        res = []
        self.model.eval()

        for instance in tqdm(requests, desc="Loglikelihood", leave=False):
            context, continuation = instance.args
            ctx_enc = self.tokenizer.encode(context, add_special_tokens=False)
            cont_enc = self.tokenizer.encode(continuation, add_special_tokens=False)

            if len(cont_enc) == 0:
                # Empty continuation has log-likelihood 0
                res.append((0.0, False))
                continue

            # Create full sequence
            full_enc = ctx_enc + cont_enc
            if len(full_enc) > self.max_length:
                full_enc = full_enc[-self.max_length :]
                ctx_enc = full_enc[: -len(cont_enc)]

            tokens = torch.tensor([full_enc], device=self.device)

            with torch.no_grad():
                timesteps = torch.zeros(1, device=self.device, dtype=torch.long)
                logits = self.model(tokens, timesteps)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Align logits: logits[i] predicts token[i+1]
                # For continuation starting at position len(ctx_enc):
                # We need logits from positions [len(ctx_enc)-1 : len(ctx_enc)+len(cont_enc)-1]
                start_idx = len(ctx_enc) - 1
                end_idx = start_idx + len(cont_enc)

                # Enforce bounds
                if start_idx < 0 or end_idx > len(full_enc):
                    res.append((float("-inf"), False))
                    continue

                logits_cont = log_probs[:, start_idx:end_idx, :]
                cont_tokens = torch.tensor([cont_enc], device=self.device)

                selected_log_probs = torch.gather(
                    logits_cont, 2, cont_tokens.unsqueeze(-1)
                ).squeeze(-1)

                log_likelihood = selected_log_probs.sum().item()

            res.append((log_likelihood, False))

        return res

    def loglikelihood_rolling(self, requests):
        res = []
        self.model.eval()

        for instance in tqdm(requests, desc="Rolling Loglikelihood", leave=False):
            # Rolling uses (context,) tuples, not (context, continuation)
            (string,) = instance.args
            enc = self.tokenizer.encode(string, add_special_tokens=False)

            if len(enc) == 0:
                res.append((0.0, False))
                continue

            # Truncate if needed
            if len(enc) > self.max_length:
                enc = enc[: self.max_length]

            tokens = torch.tensor([enc], device=self.device)

            with torch.no_grad():
                # Get predictions at t=0
                timesteps = torch.zeros(1, device=self.device, dtype=torch.long)
                logits = self.model(tokens, timesteps)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Compute log-likelihood for all tokens except the first
                # (first token has no preceding context to predict from)
                target_tokens = tokens[:, 1:]  # Tokens to predict
                pred_log_probs = log_probs[:, :-1, :]  # Predictions for those tokens

                selected_log_probs = torch.gather(
                    pred_log_probs, 2, target_tokens.unsqueeze(-1)
                ).squeeze(-1)

                log_likelihood = selected_log_probs.sum().item()

            res.append((log_likelihood, False))

        return res

    def generate_until(self, requests):
        res = []
        self.model.eval()

        for context, gen_kwargs in tqdm(requests, desc="Generate Until", leave=False):
            tokens = self.tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - 100,
            )

            max_gen_tokens = gen_kwargs.get("max_gen_toks", 100)
            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            generated = self._generate(
                tokens.input_ids.to(self.device), max_tokens=max_gen_tokens
            )

            text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            for stop in until:
                if stop in text:
                    text = text[: text.index(stop)]

            res.append(text)

        return res

    def _generate(self, context_ids, max_tokens=100):
        # Generate tokens using diffusion denoising + autoregressive extension
        self.model.eval()
        with torch.no_grad():
            B, context_len = context_ids.shape

            current_ids = torch.cat(
                [
                    context_ids,
                    torch.randint(
                        0, self.model.vocab_size, (B, max_tokens), device=self.device
                    ),
                ],
                dim=1,
            )

            for t in reversed(range(self.diffusion_steps)):
                timesteps = torch.full((B,), t, device=self.device, dtype=torch.long)
                logits = self.model(current_ids, timesteps)

                # Only update the generation region, keep context fixed
                gen_logits = logits[:, context_len:]

                if t > 0:
                    probs = torch.softmax(gen_logits, dim=-1)
                    next_tokens = torch.multinomial(
                        probs.reshape(-1, probs.size(-1)), num_samples=1
                    ).reshape(B, -1)
                else:
                    next_tokens = torch.argmax(gen_logits, dim=-1)

                current_ids = torch.cat(
                    [current_ids[:, :context_len], next_tokens], dim=1
                )

        return current_ids

    def _denoise_step(self, noisy_ids, timestep):
        self.model.eval()
        with torch.no_grad():
            noisy_ids = noisy_ids.to(self.device)
            B = noisy_ids.shape[0]
            timesteps = torch.full((B,), timestep, device=self.device, dtype=torch.long)
            logits = self.model(noisy_ids, timesteps)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CRUX model using lm-eval")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )

    args, _ = parser.parse_known_args()

    import yaml

    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    def get_default(key, section="training", default=None):
        if section in config and key in config[section]:
            return config[section][key]
        return default

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/bench",
        help="Directory for output results",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_easy,hellaswag,winogrande",  # Simplified default
        help="Comma-separated list of evaluation tasks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=get_default("diffusion_steps", section="model", default=32),
        help="Number of diffusion steps for generation",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=get_default("hidden_size", section="model", default=384),
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=get_default("intermediate_size", section="model", default=768),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=get_default("num_layers", section="model", default=10),
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=get_default("num_heads", section="model", default=8),
    )
    parser.add_argument(
        "--recursive-depth",
        type=int,
        default=get_default("recursive_depth", section="model", default=4),
    )
    parser.add_argument(
        "--use-mask-token",
        action="store_true",
        default=get_default("use_mask_token", section="model", default=False),
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=get_default("snr_min", section="model", default=-9.0),
    )
    parser.add_argument(
        "--snr-max",
        type=float,
        default=get_default("snr_max", section="model", default=9.0),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CRUX Model Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Device: {device}")
    print("=" * 70)

    print("\n[1/3] Loading tokenizer...")
    from data import get_tokenizer

    tokenizer = get_tokenizer()

    vocab_size = len(tokenizer)
    print(f"\tVocab size: {vocab_size}")

    # First load checkpoint to detect architecture
    print(f"\tLoading checkpoint {checkpoint_path.name} to detect architecture...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Detect if it's Heavy (has .layers. in keys), Light (has .shared_middle.), or Ablation (has .layers. but 4 of them)
    is_heavy = any("layers." in k for k in state_dict.keys())

    if is_heavy:
        # Check if it's the 4-layer ablation or the 10-layer heavy
        import re

        layer_indices = [
            int(re.search(r"layers\.(\d+)\.", k).group(1))
            for k in state_dict.keys()
            if re.search(r"layers\.(\d+)\.", k)
        ]
        max_layer = max(layer_indices) if layer_indices else 0

        if max_layer < 5:  # Ablation has 4 layers (0-3)
            model_type = "Ablation"
            from model_ablation import Crux as CruxAblation

            model_class = CruxAblation
        else:
            model_type = "Heavy"
            from model_heavy import Crux as CruxHeavy

            model_class = CruxHeavy
    else:
        model_type = "Light"
        from model import Crux as CruxLight

        model_class = CruxLight

    print(f"\tDetected model architecture: {model_type}")

    print(f"\n[2/3] Initializing {model_type} model...")
    model = model_class(
        vocab_size=vocab_size,
        max_seq_len=512,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        diffusion_steps=args.diffusion_steps,
        recursive_depth=args.recursive_depth,
        use_mask_token=args.use_mask_token,
        dropout=0.0,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
    )

    # Remove torch.compile artifacts if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned_state_dict[k[10:]] = v
        elif k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
        print("\tWeights loaded successfully")
        if "step" in checkpoint:
            print(f"\tCheckpoint step: {checkpoint['step']}")
    except Exception as e:
        print(f"\tError loading weights: {e}")
        sys.exit(1)

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\tParameters: {num_params:,}")

    crux_hflm = CruxHFLM(
        model=model, tokenizer=tokenizer, diffusion_steps=args.diffusion_steps
    )

    tasks = [t.strip() for t in args.tasks.split(",")]
    print("\n[3/3] Running evaluation...")
    print(f"   Tasks: {', '.join(tasks)}")
    if args.limit:
        print(f"   Limit: {args.limit} examples per task")
    print()

    results = simple_evaluate(
        model=crux_hflm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=1,
        device=device,
        limit=args.limit,
    )

    print("\n" + "=" * 70)

    table = utils.make_table(results)
    print(table)

    results_path = output_dir / "bench_results.json"
    table_path = output_dir / "bench_results_table.txt"

    import json

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(table_path, "w") as f:
        f.write(table)

    print("\n" + "=" * 70)
    print(f"Results saved to: {results_path}")
    print(f"Table saved to: {table_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

##
# Quick test (limited examples):
# python bench.py --checkpoint results/checkpoint_40000.pt --tasks arc_easy --limit 100
##
# Full evaluation (default tasks):
# python bench.py --checkpoint results/checkpoint_40000.pt
##
# Custom tasks:
# python bench.py --checkpoint results/checkpoint_40000.pt \
#     --tasks "arc_easy,arc_challenge,hellaswag,winogrande,lambada_openai"
##
# With specific GPU:
# CUDA_VISIBLE_DEVICES=2 python bench.py --checkpoint results/checkpoint_40000.pt
##
