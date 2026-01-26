import os
import json
import yaml
import torch
import wandb
import argparse
import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from pathlib import Path
from typing import Optional
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Crux, CautiousWeightDecay
from data import get_tokenizer, create_datasets, create_dataloader
from utils import save_checkpoint, load_checkpoint, set_seeds, count_params


@torch.compile
def zeropower(grad_mat, steps=5, eps=1e-7):
    # Orthogonalize G using iterative Newton's method
    shape = grad_mat.shape
    if len(shape) < 2:
        return grad_mat
    norm_mat = grad_mat.reshape(shape[0], -1)
    if norm_mat.shape[0] > norm_mat.shape[1]:
        norm_mat = norm_mat.T
    norm_mat = norm_mat / (norm_mat.norm() + eps)
    for _ in range(steps):
        norm_mat = 1.5 * norm_mat - 0.5 * norm_mat @ norm_mat.T @ norm_mat
    return norm_mat.reshape(shape)


class MuonCompositeOptimizer:
    # https://kellerjordan.github.io/posts/muon/
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 0.01,
        lr_norm_bias: float = 6e-4,
        weight_decay: float = 0.05,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.lr_norm_bias = lr_norm_bias
        self.weight_decay = weight_decay
        self.betas = betas

        muon_params, adamw_params = [], []

        for name, p in model.named_parameters():
            if p.ndim >= 2 and any(
                k in name
                for k in [
                    "layers",
                    "recursive",
                    "diffusion_head",
                    "adapter",
                    "colbert_head",
                ]
            ):
                muon_params.append(p)
            else:
                adamw_params.append(p)

        self.muon_optimizer = {
            "params": muon_params,
            "lr": lr,
            "moment": [torch.zeros_like(p) for p in muon_params],
        }

        # SGD optim for Muon params (makes scaler.unscale_ work)
        # Applied selectively to transformer layers, recursive layers, diffusion head
        self.muon_optim_heavy = torch.optim.SGD(muon_params, lr=lr)

        # AdamW for biases, embeddings, layer norm
        self.adamw_optim_util = torch.optim.AdamW(
            adamw_params,
            lr=lr_norm_bias,
            betas=betas,
            eps=eps,
            weight_decay=0,
            fused=torch.cuda.is_available(),
        )

        self.cautious_wd = CautiousWeightDecay(weight_decay=weight_decay)
        self.param_groups = self.adamw_optim_util.param_groups + [
            {"params": muon_params, "lr": lr}
        ]

    def step(self, *args, **kwargs):
        # store old weights for cautious WD track
        old_params = {}
        for p in self.muon_optimizer["params"]:
            old_params[id(p)] = p.data.clone()
        for group in self.adamw_optim_util.param_groups:
            for p in group["params"]:
                old_params[id(p)] = p.data.clone()

        # step muon
        for i, p in enumerate(self.muon_optimizer["params"]):
            if p.grad is None:
                continue

            m = self.muon_optimizer["moment"][i]
            m.mul_(self.betas[0]).add_(p.grad, alpha=1.0 - self.betas[0])
            update = zeropower(m)
            
            p.data.add_(
                update,
                alpha=-self.muon_optimizer["lr"]
                * max(1, p.shape[0] / p.shape[1]) ** 0.5,
            )

        # adamw step
        self.adamw_optim_util.step()
        # cautious wd
        all_params, all_grads, all_updates = [], [], []
        # combine params from both optimizers
        for p in self.muon_optimizer["params"]:
            if p.grad is not None:
                all_params.append(p)
                all_grads.append(p.grad)
                all_updates.append(p.data - old_params[id(p)])
        for group in self.adamw_optim_util.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    all_params.append(p)
                    all_grads.append(p.grad)
                    all_updates.append(p.data - old_params[id(p)])
        # apply cautious wd to the combined params
        self.cautious_wd.apply(all_params, all_grads, all_updates, lr=self.lr_norm_bias)

    def zero_grad(self):
        self.adamw_optim_util.zero_grad()
        self.muon_optim_heavy.zero_grad()
        for p in self.muon_optimizer["params"]:
            p.grad = None

    def state_dict(self):
        return {
            "adamw": self.adamw_optim_util.state_dict(),
            "muon_moment": self.muon_optimizer["moment"],
        }

    def load_state_dict(self, state_dict):
        self.adamw_optim_util.load_state_dict(state_dict["adamw"])
        self.muon_optimizer["moment"] = state_dict["muon_moment"]


class SchedulerWithMinLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio

        self.adamw_optimizer = optimizer.adamw_optim_util
        self.muon_optim = optimizer.muon_optimizer

        self.base_adamw_lr = self.adamw_optimizer.param_groups[0]["lr"]
        self.base_muon_lr = self.muon_optim["lr"]

        self._step_count = 0

    def get_lr(self):
        step = self._step_count
        if step < self.num_warmup_steps and self.num_warmup_steps > 0:
            factor = step / self.num_warmup_steps
        elif self.min_lr_ratio >= 1.0:
            factor = 1.0
        else:
            steps_since_warmup = step - self.num_warmup_steps
            total_decay_steps = self.num_training_steps - self.num_warmup_steps
            if total_decay_steps <= 0:
                factor = 1.0
            else:
                progress = min(1.0, steps_since_warmup / total_decay_steps)
                cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        return self.base_muon_lr * factor

    def step(self):
        self._step_count += 1
        factor = self.get_lr() / self.base_muon_lr if self.base_muon_lr > 0 else 1.0

        # update AdamW LR
        for param_group in self.adamw_optimizer.param_groups:
            param_group["lr"] = self.base_adamw_lr * factor

        # update Muon LRs
        self.muon_optim["lr"] = self.base_muon_lr * factor
        # update as well for unscale_ consistency (though speed doesn't matter)
        # self.muon_optim.param_groups[0]['lr'] = self.muon_optim['lr']

    def state_dict(self):
        return {"_step_count": self._step_count}

    def load_state_dict(self, state_dict):
        self._step_count = state_dict["_step_count"]


def is_main_rank(rank: int) -> bool:
    return rank == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def setup_distributed(local_rank: int):
    # Check if we're in a distributed environment (torchrun sets these)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True
    return False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(
    model,
    train_loader,
    val_loader,
    output_dir: str,
    num_steps: int = 100000,
    learning_rate: float = 2e-4,
    learning_rate_norm_bias: float = 7.5e-5,
    warmup_steps: int = 4000,
    grad_clip: float = 0.75,
    weight_decay: float = 0.05,
    eval_interval: int = 1000,
    save_interval: int = 5000,
    log_interval: int = 100,
    device: str = "cuda",
    resume_from: Optional[str] = None,
    min_lr_ratio: float = 0.1,
    rank: int = 0,
    world_size: int = 1,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
):
    optimizer = MuonCompositeOptimizer(
        model=model,
        lr=learning_rate,
        lr_norm_bias=learning_rate_norm_bias,
        weight_decay=weight_decay,
    )

    scheduler = SchedulerWithMinLR(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps // grad_accum_steps,
        num_training_steps=num_steps // grad_accum_steps,
        min_lr_ratio=min_lr_ratio,
    )

    start_step = 0
    if resume_from:
        start_step = load_checkpoint(
            Path(resume_from),
            model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if is_main_rank(rank):
            print(f"Resumed from step {start_step}")

    log_dir = Path(output_dir) / "logs"
    if is_main_rank(rank):
        log_dir.mkdir(exist_ok=True, parents=True)

    use_amp_training = use_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp_training)
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if is_main_rank(rank):
        if use_amp_training:
            print(f"\tUsing AMP with dtype: {autocast_dtype}")
        print(f"\tGradient accumulation steps: {grad_accum_steps}")
        print(
            f"\tEffective batch size: {train_loader.batch_size * grad_accum_steps * world_size}"
        )

    model.train()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_iter = iter(train_loader)
    loss_accum = 0.0
    micro_step = 0

    pbar = tqdm(range(start_step, num_steps), desc="Training", disable=rank != 0)

    for step in pbar:
        # Reset train_iter once exhausted
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch.input_ids.to(device)
        model_to_call = model.module if isinstance(model, DDP) else model

        with torch.amp.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=use_amp_training
        ):
            loss = model_to_call.diffusion_loss(input_ids) / grad_accum_steps

        scaler.scale(loss).backward()
        loss_accum += loss.item() * grad_accum_steps
        micro_step += 1

        # step only every grad_accum_steps
        if micro_step % grad_accum_steps == 0:
            scaler.unscale_(optimizer.adamw_optim_util)
            scaler.unscale_(optimizer.muon_optim_heavy)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # derive debug metrics before zero_grad
            rec_norm = 0.0
            embed_norm = 0.0
            if is_main_rank(rank):
                with torch.no_grad():
                    for p in model_to_call.recursive.parameters():
                        if p.grad is not None:
                            rec_norm += p.grad.norm(2).item() ** 2
                    rec_norm = rec_norm**0.5

                    for p in model_to_call.token_embedding.parameters():
                        if p.grad is not None:
                            embed_norm += p.grad.norm(2).item() ** 2
                    embed_norm = embed_norm**0.5

            # composite optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if is_main_rank(rank):
                wandb.log(
                    {
                        "train/loss": loss_accum / grad_accum_steps,
                        "train/lr_weight": optimizer.muon_optimizer["lr"],
                        "train/lr_norm": optimizer.adamw_optim_util.param_groups[0][
                            "lr"
                        ],
                        "train/grad_norm": grad_norm.item()
                        if isinstance(grad_norm, torch.Tensor)
                        else grad_norm,
                        "train/grad_norm_recursive": rec_norm,
                        "train/grad_norm_embed": embed_norm,
                    },
                    step=step,
                )

            loss_accum = 0.0

        # if eval_interval is met... evaluate
        if step % eval_interval == 0 and val_loader is not None:
            if world_size > 1:
                dist.barrier()

            val_loss, val_perplexity = evaluate(
                model, val_loader, device, rank, world_size
            )

            if is_main_rank(rank):
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/perplexity": val_perplexity,
                    },
                    step=step,
                )

                pbar.write(
                    f"Step {step}: val_loss={val_loss:.4f}, val_perplexity={val_perplexity:.2f}"
                )

                val_entry = {
                    "step": step,
                    "val_loss": val_loss,
                    "val_perplexity": val_perplexity,
                }

                with open(log_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(val_entry) + "\n")

            model.train()

        if step % save_interval == 0 and step > 0:
            if world_size > 1:
                dist.barrier()
            if is_main_rank(rank):
                save_path = Path(output_dir) / f"checkpoint_{step}.pt"
                # unwrap DDP model, then save
                model_to_save = model.module if isinstance(model, DDP) else model
                save_checkpoint(model_to_save, optimizer, scheduler, step, save_path)

    if world_size > 1:
        dist.barrier()

    if is_main_rank(rank):
        save_path = Path(output_dir) / "final.pt"
        # unwrap DDP model, then save
        model_to_save = model.module if isinstance(model, DDP) else model
        save_checkpoint(model_to_save, optimizer, scheduler, num_steps, save_path)

        with open(log_dir / "metrics.jsonl", "a") as f:
            json.dump({"training_complete": True, "final_step": num_steps}, f)
            f.write("\n")

    wandb.finish()


def evaluate(
    model, val_loader, device: str = "cuda", rank: int = 0, world_size: int = 1
):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # access underlying model if DDP-wrapped
    model_to_call = model.module if isinstance(model, DDP) else model

    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc="Evaluating", leave=False, disable=rank != 0
        ):
            input_ids = batch.input_ids.to(device)
            loss = model_to_call.diffusion_loss(input_ids)
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    if world_size > 1:
        # convert to tensors to go through distributed reduction
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_tokens_tensor = torch.tensor(total_tokens, device=device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)

        total_loss = total_loss_tensor.item()
        total_tokens = total_tokens_tensor.item()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRUX model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )

    args, _ = parser.parse_known_args()

    try:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
        else:
            config = None
            print(f"Config file {config_path} not found, using defaults")
    except ImportError:
        config = None
        print("Warning: pyyaml not installed, using CLI defaults")

    def get_default(key, section="training"):
        if config and section in config and key in config[section]:
            return config[section][key]
        return None

    parser.add_argument(
        "--data-path",
        type=str,
        default=get_default("path", section="data") or "data/MiniPile_DensityPico",
        help="Path to dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=get_default("dataset", section="data"),
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=get_default("output_dir", section="training") or "results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=get_default("num_steps", section="training") or 100000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=get_default("batch_size", section="data") or 8,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=get_default("seq_len", section="data") or 512,
        help="Sequence length",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=get_default("learning_rate", section="training") or 1e-3,
        help="Learning rate for weights",
    )
    parser.add_argument(
        "--learning-rate-norm-bias",
        type=float,
        default=get_default("learning_rate_norm_bias", section="training") or 0.006,
        help="Learning rate for norms/biases",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=get_default("warmup_steps", section="training") or 2000,
        help="Warmup steps",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=get_default("grad_clip", section="training") or 1.0,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=get_default("weight_decay", section="training") or 0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=get_default("eval_interval", section="training") or 100,
        help="Evaluation interval",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=get_default("save_interval", section="training") or 1000,
        help="Checkpoint save interval",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=get_default("log_interval", section="training") or 10,
        help="Logging interval",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed training (auto-set by torchrun)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--recursive-depth",
        type=int,
        default=get_default("recursive_depth", section="model") or 4,
        help="Depth of recursive processing (default: 4)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=get_default("hidden_size", section="model") or 384,
        help="Hidden size (default: 384)",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=get_default("intermediate_size", section="model") or 768,
        help="Intermediate size (default: 768)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=get_default("num_layers", section="model") or 10,
        help="Number of transformer layers (default: 10)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=get_default("num_heads", section="model") or 8,
        help="Number of attention heads (default: 8)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=get_default("dropout", section="model") or 0.1,
        help="Dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=get_default("diffusion_steps", section="model") or 16,
        help="Number of diffusion steps (default: 16)",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=get_default("snr_min", section="model") or -9.0,
        help="Minimum SNR (default: -9.0)",
    )
    parser.add_argument(
        "--snr-max",
        type=float,
        default=get_default("snr_max", section="model") or 9.0,
        help="Maximum SNR (default: 9.0)",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=get_default("min_lr_ratio", section="training") or 0.1,
        help="Minimum learning rate ratio (default: 0.1)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=get_default("compile", section="training") or False,
        help="Use torch.compile for model optimization (default: False)",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=get_default("use_amp", section="training") or False,
        help="Enable automatic mixed precision training",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=get_default("grad_accum_steps", section="training") or 1,
        help="Gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=get_default("gradient_checkpointing", section="training") or False,
        help="Enable gradient checkpointing to reduce memory usage",
    )
    parser.add_argument(
        "--use-mask-token",
        action="store_true",
        default=get_default("use_mask_token", section="model") or False,
        help="Use a dedicated mask token instead of random tokens during noising",
    )

    args = parser.parse_args()

    set_seeds(42)
    output_dir = Path(args.output_dir)
    if args.local_rank == 0:
        output_dir.mkdir(exist_ok=True, parents=True)

    # Setup distributed training
    is_distributed = setup_distributed(args.local_rank)

    if is_distributed:
        rank = get_rank()
        world_size = get_world_size()
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)

    if is_main_rank(rank):
        print(f"World size: {world_size}, Rank: {rank}, Distributed: {is_distributed}")

    tokenizer = get_tokenizer()

    model = Crux(
        vocab_size=len(tokenizer),
        max_seq_len=args.seq_len,
        use_mask_token=args.use_mask_token,
        recursive_depth=args.recursive_depth,
        gradient_checkpointing=args.gradient_checkpointing,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        diffusion_steps=args.diffusion_steps,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
    )

    if is_main_rank(rank):
        print(f"Parameters: {count_params(model):,}")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.compile:
        if is_main_rank(rank):
            print("Compiling model with torch.compile (mode=reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")

    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    if is_main_rank(rank):
        print("Loading datasets...")
    train_dataset, val_dataset = create_datasets(
        tokenizer=tokenizer,
        data_path=args.data_path,
        seq_len=args.seq_len,
        hf_dataset_name=args.dataset,
    )
    if is_main_rank(rank):
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    # Pass rank and world_size to dataloader
    train_loader = create_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        num_replicas=world_size,
        rank=rank,
    )
    val_loader = create_dataloader(
        dataset=val_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        num_replicas=world_size,
        rank=rank,
    )

    if is_main_rank(rank):
        print("Configuration:")
        for arg in vars(args):
            print(f"\t{arg}: {getattr(args, arg)}")

    wandb.init(
        project="crux-diffusion-lm",
        name="run-debug-001",
        config={
            "model": "BlockDiffusionTransformer",
            "learning_rate": args.learning_rate,
            "min_lr_ratio": args.min_lr_ratio,
            "warmup_steps": args.warmup_steps,
            "batch_size": args.batch_size,
            "grad_clip": args.grad_clip,
            "diffusion_steps": args.diffusion_steps,
            "snr_min": args.snr_min,
            "snr_max": args.snr_max,
        },
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        learning_rate_norm_bias=args.learning_rate_norm_bias,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        device=device,
        resume_from=args.resume_from,
        min_lr_ratio=args.min_lr_ratio,
        rank=rank,
        world_size=world_size,
        use_amp=args.use_amp,
        grad_accum_steps=args.grad_accum_steps,
    )

    cleanup_distributed()

# conda create --name crux python=3.11 -y
# conda activate crux
# pip install -r requirements.txt

# Single GPU (plain python):
# CUDA_VISIBLE_DEVICES=2 python src/train.py
# Multi-GPU with torchrun:
# torchrun --nproc_per_node=2 train.py
# Multi-GPU with torchrun (4 GPUs):
# torchrun --nproc_per_node=4 train.py
# To see detailed error traces with torchrun, set:
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# or use: TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=4 train.py

# tmux new -s crux
# conda activate crux
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t crux
# tmux list-sessions
# tmux kill-session -t crux
