import math
import torch
import torch.nn as nn

from typing import Optional
from torch.utils.checkpoint import checkpoint


class CautiousWeightDecay:
    def __init__(self, weight_decay: float = 0.01):
        self.weight_decay = weight_decay

    def apply(self, params, grads, updates, lr: float = 1.0):
        for param, _, update in zip(params, grads, updates):
            if param.ndim < 2:
                continue
            mask = (update * param > 0).float()
            # lr scales weight decay
            param.data.mul_(1 - self.weight_decay * lr * mask)


class BlockDiffusionAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, block_size: int, dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads  # Attn head count
        self.block_size = block_size  # Size of blocks for block-wise attn
        self.head_dim = hidden_size // num_heads  # Partial dimension per head

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.fin_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def create_vectorized_mask(self, q_len: int, k_len: int, device: torch.device):
        # Bidirectional attention, see the whole noisy sequence
        return torch.ones(q_len, k_len, device=device, dtype=torch.bool)

    def forward(self, x, context=None):
        B, L, D = x.shape
        if context is not None:
            full_inp = torch.cat([x, context], dim=1)
            total_L = full_inp.shape[1]
            q = (
                self.q_proj(full_inp)
                .view(B, total_L, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                self.k_proj(full_inp)
                .view(B, total_L, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(full_inp)
                .view(B, total_L, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            mask = self.create_vectorized_mask(total_L, total_L, x.device)
        else:
            q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))

        attn_mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = self.fin_proj(out.transpose(1, 2).reshape(B, -1, D))

        if context is not None:
            return out[:, :L], out[:, L:]
        return out


class RecursiveProcessing(nn.Module):
    def __init__(self, hidden_size: int, depth: int = 2):
        super().__init__()
        self.depth = depth
        self.gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.agg = nn.Linear(hidden_size * 2, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.time_gate = nn.Linear(hidden_size, hidden_size)
        self.beta = nn.Parameter(torch.ones(1) * -4.0)

    def forward(self, x, context, t_emb: Optional[torch.Tensor] = None):
        res = x
        beta_mod = 1.0
        if t_emb is not None:
            gate_mod = torch.sigmoid(self.time_gate(t_emb)).unsqueeze(1)
            beta_mod = gate_mod

        for _ in range(self.depth):
            x_norm = self.ln(x)
            g = self.gate(torch.cat([x_norm, context], dim=-1))
            transformed = self.transform(x_norm)
            x = self.agg(torch.cat([g * transformed + (1 - g) * x, context], dim=-1))

        beta = torch.sigmoid(self.beta) * beta_mod

        # Standard residual for more direct gradient flow, follow with norm later
        return (res + beta * x) / (1.0 + beta)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = BlockDiffusionAttention(hidden_size, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        if context is not None:
            res_x, res_c = x, context
            x, context = self.attn(self.ln1(x), self.ln1(context))
            x = res_x + self.dropout(x)
            context = res_c + self.dropout(context)
            x = x + self.dropout(self.ffn(self.ln2(x)))
            context = context + self.dropout(self.ffn(self.ln2(context)))
            return x, context
        else:
            x = x + self.dropout(self.attn(self.ln1(x)))
            x = x + self.dropout(self.ffn(self.ln2(x)))
            return x


class Crux(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_len: int = 512,
        hidden_size: int = 384,
        intermediate_size: int = 768,
        num_layers: int = 10,
        num_heads: int = 8,
        diffusion_steps: int = 16,
        dropout: float = 0.1,
        snr_min: float = -9.0,
        snr_max: float = 9.0,
        use_mask_token: bool = False,
        recursive_depth: int = 4,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.use_mask_token = use_mask_token
        self.mask_token_id = vocab_size if use_mask_token else None
        self.actual_vocab_size = vocab_size + 1 if use_mask_token else vocab_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.diffusion_steps = diffusion_steps

        self.token_embedding = nn.Embedding(self.actual_vocab_size, hidden_size)
        self.positional_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.attention_sinks = nn.Parameter(torch.zeros(4, hidden_size))

        # Looped-Core Architecture: Entry, Shared Middle, and Exit
        self.entry_layer = TransformerLayer(
            hidden_size, intermediate_size, num_heads, 64, dropout
        )
        self.shared_middle = TransformerLayer(
            hidden_size, intermediate_size, num_heads, 64, dropout
        )
        self.exit_layer = TransformerLayer(
            hidden_size, intermediate_size, num_heads, 64, dropout
        )
        self.num_middle_loops = 8  # Configurable outer loops

        self.recursive = RecursiveProcessing(hidden_size, depth=recursive_depth)
        self.recursive_norm = nn.LayerNorm(hidden_size)  # New stability norm
        self.output_norm = nn.LayerNorm(hidden_size)

        self.diffusion_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=True),
        )

        self.colbert_dim = 128
        self.colbert_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, self.colbert_dim, bias=True),
        )

        self.dropout = nn.Dropout(dropout)
        self.register_diffusion_schedule(snr_min=snr_min, snr_max=snr_max)
        self._init_weights()

    def _init_weights(self):
        # Embeddings - use smaller std for stability
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding.weight, std=0.02)
        nn.init.normal_(self.attention_sinks, std=0.02)

        # Time embedding
        nn.init.normal_(self.time_embed[0].weight, std=0.02)
        nn.init.zeros_(self.time_embed[0].bias)
        nn.init.normal_(self.time_embed[2].weight, std=0.02 / 2)
        nn.init.zeros_(self.time_embed[2].bias)

        # Transformer layers (Sandwich: Entry, Shared Middle, Exit)
        for name, layer in [
            ("entry", self.entry_layer),
            ("shared", self.shared_middle),
            ("exit", self.exit_layer),
        ]:
            depth_scale = 1.0 / math.sqrt(2 * (1 + self.num_middle_loops + 1))
            # Attention
            nn.init.normal_(layer.attn.q_proj.weight, std=0.02)
            nn.init.normal_(layer.attn.k_proj.weight, std=0.02)
            nn.init.normal_(layer.attn.v_proj.weight, std=0.02)
            nn.init.zeros_(layer.attn.q_proj.bias)
            nn.init.zeros_(layer.attn.k_proj.bias)
            nn.init.zeros_(layer.attn.v_proj.bias)
            nn.init.normal_(layer.attn.fin_proj.weight, std=0.02 * depth_scale)
            nn.init.zeros_(layer.attn.fin_proj.bias)
            # FFN
            nn.init.normal_(layer.ffn[0].weight, std=0.02)
            nn.init.zeros_(layer.ffn[0].bias)
            nn.init.normal_(layer.ffn[2].weight, std=0.02 * depth_scale)
            nn.init.zeros_(layer.ffn[2].bias)
            # LayerNorms
            nn.init.ones_(layer.ln1.weight)
            nn.init.zeros_(layer.ln1.bias)
            nn.init.ones_(layer.ln2.weight)
            nn.init.zeros_(layer.ln2.bias)

        # Recursive processing
        recursive_scale = 0.02 / math.sqrt(self.recursive.depth)
        nn.init.normal_(self.recursive.gate[0].weight, std=recursive_scale)
        nn.init.zeros_(self.recursive.gate[0].bias)
        nn.init.normal_(self.recursive.transform.weight, std=recursive_scale)
        nn.init.zeros_(self.recursive.transform.bias)
        nn.init.normal_(self.recursive.agg.weight, std=recursive_scale)
        nn.init.zeros_(self.recursive.agg.bias)
        nn.init.normal_(self.recursive.time_gate.weight, std=recursive_scale)
        nn.init.zeros_(self.recursive.time_gate.bias)
        nn.init.ones_(self.recursive.ln.weight)
        nn.init.zeros_(self.recursive.ln.bias)
        nn.init.ones_(self.recursive_norm.weight)
        nn.init.zeros_(self.recursive_norm.bias)

        # Output norm
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)

        # Diffusion head - smaller init
        nn.init.normal_(self.diffusion_head[0].weight, std=0.02)
        nn.init.zeros_(self.diffusion_head[0].bias)
        nn.init.normal_(
            self.diffusion_head[2].weight, std=0.01
        )  # Even smaller for final layer
        nn.init.zeros_(self.diffusion_head[2].bias)

        # Adapter
        nn.init.normal_(self.adapter[0].weight, std=0.02)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.normal_(self.adapter[2].weight, std=0.02 / 2)
        nn.init.zeros_(self.adapter[2].bias)

        # ColBERT head
        nn.init.normal_(self.colbert_head[0].weight, std=0.02)
        nn.init.zeros_(self.colbert_head[0].bias)
        nn.init.normal_(self.colbert_head[2].weight, std=0.02)
        nn.init.zeros_(self.colbert_head[2].bias)

    def register_diffusion_schedule(self, snr_min: float, snr_max: float):
        T = self.diffusion_steps
        snr = torch.linspace(snr_max, snr_min, T)
        alphas = torch.sigmoid(snr)
        alphas = torch.clamp(alphas, min=1e-4, max=1 - 1e-4)
        self.register_buffer("alphas_cumprod", alphas)

    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.hidden_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(
        self, input_ids: torch.Tensor, timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.token_embedding(input_ids)
        x = x + self.positional_embedding(torch.arange(L, device=x.device).unsqueeze(0))

        if timesteps is not None:
            t_emb = self.get_timestep_embedding(timesteps)
            t_emb = t_emb.to(x.dtype)
            t_emb = self.time_embed(t_emb)
            x = x + t_emb.unsqueeze(1)

        sinks = self.attention_sinks.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([sinks, x], dim=1)

        context = x.clone()

        # 1. Entry Layer
        if self.gradient_checkpointing and self.training:
            x, context = checkpoint(self.entry_layer, x, context, use_reentrant=False)
        else:
            x, context = self.entry_layer(x, context)

        # 2. Looped Middle Block (8 passes)
        for _ in range(self.num_middle_loops):
            if self.gradient_checkpointing and self.training:
                x, context = checkpoint(
                    self.shared_middle, x, context, use_reentrant=False
                )
            else:
                x, context = self.shared_middle(x, context)

            # Recursive processing nested inside the loop
            processed_rec = self.recursive(
                x[:, 4:], context[:, 4:], t_emb=t_emb if timesteps is not None else None
            )
            x = torch.cat([x[:, :4], self.recursive_norm(processed_rec)], dim=1)
            context = x.clone()

        # 3. Exit Layer
        if self.gradient_checkpointing and self.training:
            x, context = checkpoint(self.exit_layer, x, context, use_reentrant=False)
        else:
            x, context = self.exit_layer(x, context)

        x = x[:, 4:]  # Remove sinks
        x = self.output_norm(x)
        x = self.diffusion_head(x)

        # Residual adapter with smaller contribution
        x = x + 0.1 * self.adapter(x)  # Scale down adapter contribution
        # Weight tying
        return nn.functional.linear(x, self.token_embedding.weight[: self.vocab_size])

    def get_colbert_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.token_embedding(input_ids)
        x = x + self.positional_embedding(torch.arange(L, device=x.device).unsqueeze(0))
        sinks = self.attention_sinks.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([sinks, x], dim=1)
        context = x.clone()

        # 1. Entry
        x, context = self.entry_layer(x, context)

        # 2. Looped Middle
        for _ in range(self.num_middle_loops):
            x, context = self.shared_middle(x, context)
            processed_rec = self.recursive(x[:, 4:], context[:, 4:], t_emb=None)
            x = torch.cat([x[:, :4], self.recursive_norm(processed_rec)], dim=1)
            context = x.clone()

        # 3. Exit
        x, context = self.exit_layer(x, context)

        embs = self.colbert_head(x[:, 4:])
        return torch.nn.functional.normalize(embs, p=2, dim=-1)

    def diffusion_loss(self, clean_ids: torch.Tensor) -> torch.Tensor:
        B, L = clean_ids.shape
        device = clean_ids.device

        timesteps = torch.randint(0, self.diffusion_steps, (B,), device=device)
        mask_prob = 1.0 - self.alphas_cumprod[timesteps].view(B, 1)

        if self.use_mask_token:
            mask = torch.bernoulli(mask_prob.expand(B, L)).bool()
            noisy_ids = torch.where(mask, self.mask_token_id, clean_ids)
        else:
            random_tokens = torch.randint(0, self.vocab_size, (B, L), device=device)
            mask = torch.bernoulli(mask_prob.expand(B, L)).bool()
            noisy_ids = torch.where(mask, random_tokens, clean_ids)

        logits = self(noisy_ids, timesteps)
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size), clean_ids.view(-1), label_smoothing=0.1
        )
        return loss


if __name__ == "__main__":
    model = Crux()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    input_ids = torch.randint(0, 32000, (2, 64))
    logits = model(input_ids)
    print(f"Output logits shape: {logits.shape}")

    clean_ids = torch.randint(0, 32000, (2, 64))
    loss = model.diffusion_loss(clean_ids)
    print(f"Diffusion loss: {loss.item():.4f}")
