import sys
import argparse
import torch

from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import Crux
from utils import set_seeds

device = "cuda" if torch.cuda.is_available() else "cpu"


def move_cursor(n):
    sys.stdout.write(f"\033[{n}A")


def generate(model, tokenizer, prompt, num_tokens=50, seed=42):
    set_seeds(seed)

    prompt_tokens = tokenizer.encode(
        prompt, return_tensors="pt", max_length=512, truncation=True
    )
    seq_len = prompt_tokens.shape[1]
    target_len = seq_len + num_tokens

    model_device = next(model.parameters()).device
    input_device = model_device if model_device.type != "cpu" else device

    if model_device != input_device:
        model = model.to(input_device)

    print("\n#" * 70)
    print(f'\nPrompt: "{prompt}"')
    print(f"Device: {input_device}")
    print(f"Generating: {num_tokens} tokens")
    print(f"Target length: {target_len} tokens")

    current_ids = torch.randint(0, model.vocab_size, (1, target_len), device=input_device)
    current_ids[:, :seq_len] = prompt_tokens[0][:seq_len].to(input_device)

    T = model.diffusion_steps
    alphas_cumprod = model.alphas_cumprod

    print("\n=" * 70)
    print("\n[DIFFUSION DENOISING]")
    print(f"Steps: {T}")

    first_block = True
    for t in reversed(range(T)):
        noise_level = 1 - alphas_cumprod[t].item()

        with torch.no_grad():
            current_ids = current_ids.clone()
            timesteps = torch.full((1,), t, device=input_device, dtype=torch.long)

            noise_ids = torch.randint(
                0, model.vocab_size, (1, target_len), device=input_device
            )
            alpha = alphas_cumprod[t].item()
            noisy_for_model = (alpha * current_ids + (1 - alpha) * noise_ids).long()

            logits = model(noisy_for_model, timesteps)
            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probs[:, seq_len : seq_len + 1].view(-1, model.vocab_size), 1
            )
            current_ids[:, seq_len] = next_token.item()

        if t % 4 == 0 or t == 0:
            progress = ((T - 1 - t) / (T - 1)) * 100
            bar_len = 30
            filled = int(bar_len * progress / 100)
            bar = "█" * filled + "░" * (bar_len - filled)

            partial_output = tokenizer.decode(
                current_ids[0, : seq_len + 1], skip_special_tokens=True
            )

            if not first_block:
                move_cursor(5)
            first_block = False

            sys.stdout.write(
                "\r\033[K"
                + f"┌─ Step {t:2d}/{T - 1} ─┐ Noise: {noise_level:.3f} │ {bar} {progress:5.1f}%\n"
            )
            sys.stdout.write(
                "\r\033[K"
                + "├──────────────────────────────────────────────────────────────┤\n"
            )
            sys.stdout.write(
                "\r\033[K" + f'│ Output: "{partial_output[:50]:<50} │\n'
            )
            sys.stdout.write(
                "\r\033[K"
                + f"│ Token {seq_len}: {next_token.item():5d} → {tokenizer.decode([next_token.item()]):<10}\n"
            )
            sys.stdout.write(
                "\r\033[K"
                + "└──────────────────────────────────────────────────────────────┘\n"
            )
            sys.stdout.flush()

    print("=" * 70)
    print("\n[AUTOREGRESSIVE EXTENSION]")

    first_ext = True
    for i in range(target_len - seq_len):
        with torch.no_grad():
            timesteps = torch.zeros((1,), device=input_device, dtype=torch.long)
            logits = model(current_ids, timesteps)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(
                probs[:, seq_len + i : seq_len + i + 1].view(-1, model.vocab_size), 1
            )
            current_ids[:, seq_len + i] = next_token.item()

        if i % 5 == 0 or i == target_len - seq_len - 1:
            progress = (i + 1) / (target_len - seq_len) * 100
            bar_len = 30
            filled = int(bar_len * progress / 100)
            bar = "█" * filled + "░" * (bar_len - filled)

            partial_output = tokenizer.decode(current_ids[0], skip_special_tokens=True)

            if not first_ext:
                move_cursor(3)
            first_ext = False

            sys.stdout.write(
                "\r\033[K"
                + f"┌─ {i + 1:2d}/{target_len - seq_len} ─┤ {bar} {progress:5.1f}%\n"
            )
            sys.stdout.write("\r\033[K" + f"│ {partial_output[-60:]:<60} │\n")
            sys.stdout.write(
                "\r\033[K"
                + "└──────────────────────────────────────────────────────────────┘\n"
            )
            sys.stdout.flush()

    print("#" * 70)

    final_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)

    print(f'\nPrompt: "{prompt}"')
    print(f'Generated: "{final_text[len(prompt):]}"')
    print(f"Total tokens: {len(current_ids[0])}")

    return final_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with CRUX model")
    parser.add_argument("prompt", type=str, nargs="+", help="Prompt text")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--num-tokens", type=int, default=50,
                        help="Number of tokens to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    prompt = " ".join(args.prompt)

    print("\nLoading CRUX model...")
    model = Crux(vocab_size=50257, max_seq_len=512)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            cleaned_state_dict = {}
            state_dict = checkpoint["model_state_dict"]
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'): # torch.compile residual
                    name = k[10:]
                    cleaned_state_dict[name] = v
                else:
                    cleaned_state_dict[k] = v
            
            model.load_state_dict(cleaned_state_dict)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Using untrained model")
    else:
        print("No checkpoint specified, using untrained model")

    model = model.to(device)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("\nModel:")
    print(f"\tVocab: {model.vocab_size}, Steps: {model.diffusion_steps}")
    print(f"\tLayers: {model.num_layers}, Hidden: {model.hidden_size}")

    _ = generate(
        model, tokenizer, prompt,
        num_tokens=args.num_tokens,
        seed=args.seed
    )

    print("\n#" * 70)


if __name__ == "__main__":
    main()
