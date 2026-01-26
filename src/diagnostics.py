import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Optional


class Diagnostics:
    # Tracks, visualizes model training metrics to help with debugging
    
    def __init__(self, model: torch.nn.Module, log_dir: str = "results/diagnostics"):
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Track over training steps
        self.activation_histories = {}
        self.gradient_histories = {}
        self.weight_histories = {}
        self.dead_neurons = {}

    def reset(self):
        self.activation_histories = {}
        self.gradient_histories = {}
        self.weight_histories = {}
        self.dead_neurons = {}

    @torch.no_grad()
    def track_gradients(self, step: int):
        # Record gradient statistics for all model params
        gradient_stats = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                grad_max = grad.max().item()
                grad_min = grad.min().item()

                gradient_stats[name] = {
                    "norm": grad_norm,
                    "mean": grad_mean,
                    "std": grad_std,
                    "max": grad_max,
                    "min": grad_min,
                }

                if name not in self.gradient_histories:
                    self.gradient_histories[name] = []
                self.gradient_histories[name].append(
                    {
                        "step": step,
                        "norm": grad_norm,
                        "mean": grad_mean,
                        "std": grad_std,
                    }
                )

        return gradient_stats

    def track_activations(self, step: int, activations: Dict[str, torch.Tensor]):
        # Get stats on layer activations, sparsity
        activation_stats = {}

        for name, act in activations.items():
            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy()

                activation_stats[name] = {
                    "mean": float(np.mean(act)),
                    "std": float(np.std(act)),
                    "max": float(np.max(act)),
                    "min": float(np.min(act)),
                    "sparsity": float(np.mean(act == 0)),
                }

                if name not in self.activation_histories:
                    self.activation_histories[name] = []
                self.activation_histories[name].append(
                    {
                        "step": step,
                        "mean": activation_stats[name]["mean"],
                        "std": activation_stats[name]["std"],
                    }
                )

        return activation_stats

    @torch.no_grad()
    def track_weights(self, step: int):
        # Monitor weight distributions, norms across layers
        weight_stats = {}
        for name, param in self.model.named_parameters():
            if param.data is not None:
                data = param.data.cpu().numpy()
                weight_stats[name] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "max": float(np.max(data)),
                    "min": float(np.min(data)),
                    "norm": float(np.linalg.norm(data)),
                }
                if name not in self.weight_histories:
                    self.weight_histories[name] = []
                self.weight_histories[name].append(
                    {
                        "step": step,
                        "mean": weight_stats[name]["mean"],
                        "std": weight_stats[name]["std"],
                        "norm": weight_stats[name]["norm"],
                    }
                )

        return weight_stats

    def detect_dead_neurons(
        self, activations: Dict[str, torch.Tensor], threshold: float = 1e-6
    ) -> Dict[str, List[int]]:
        # Identify neurons near zero-activation
        dead_neurons = {}

        for name, act in activations.items():
            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy()

                # Average across batch for finding the consistently inactive neurons
                mean_act = np.mean(act, axis=0)
                dead_mask = np.abs(mean_act) < threshold
                dead_indices = np.where(dead_mask)[0].tolist()

                if len(dead_indices) > 0:
                    dead_neurons[name] = dead_indices

                if name not in self.dead_neurons:
                    self.dead_neurons[name] = []
                self.dead_neurons[name].append(len(dead_indices))

        return dead_neurons

    @torch.no_grad()
    def check_numerical_stability(self) -> Dict[str, bool]:
        # Verify model params; Check gradients to be numerically valid
        checks = {
            "has_nan": False,
            "has_inf": False,
            "gradients_finite": True,
        }

        for _, param in self.model.named_parameters():
            if param.data.isnan().any():
                checks["has_nan"] = True
            if param.data.isinf().any():
                checks["has_inf"] = True
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    checks["gradients_finite"] = False

        return checks

    def compute_gradient_flow_score(self) -> float:
        # Calculate gradient flow health (0-1, higher is better)
        if not self.gradient_histories:
            return 0.0

        flow_scores = []

        for _, history in self.gradient_histories.items():
            if len(history) >= 2:
                recent_grads = [h["norm"] for h in history[-10:]]
                if len(recent_grads) > 0:
                    mean_grad = np.mean(recent_grads)
                    std_grad = np.std(recent_grads)
                    # Lower coefficient of variation indicates stable gradients
                    if mean_grad > 0:
                        cv = std_grad / mean_grad
                        flow_scores.append(1.0 / (1.0 + cv))

        if flow_scores:
            return np.mean(flow_scores)
        return 0.0

    def plot_gradient_flow(self, save_path: Optional[str] = None):
        # Plots on gradient evolution over training steps
        if not self.gradient_histories:
            return

        _, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax_grad_norm = axes[0]
        for name, history in list(self.gradient_histories.items())[:10]:
            steps = [h["step"] for h in history]
            norms = [h["norm"] for h in history]
            ax_grad_norm.plot(steps, norms, label=name[:30], alpha=0.7)
        ax_grad_norm.set_xlabel("Step")
        ax_grad_norm.set_ylabel("Gradient Norm")
        ax_grad_norm.set_title("Gradient Norm Over Time")
        ax_grad_norm.legend(loc="upper right", fontsize=8)
        ax_grad_norm.grid(True, alpha=0.3)

        ax_grad_avg = axes[1]
        for name, history in list(self.gradient_histories.items())[:10]:
            steps = [h["step"] for h in history]
            means = [h["mean"] for h in history]
            ax_grad_avg.plot(steps, means, label=name[:30], alpha=0.7)
        ax_grad_avg.set_xlabel("Step")
        ax_grad_avg.set_ylabel("Gradient Mean")
        ax_grad_avg.set_title("Gradient Mean Over Time")
        ax_grad_avg.legend(loc="upper right", fontsize=8)
        ax_grad_avg.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.savefig(
                self.log_dir / "gradient_flow.png", dpi=150, bbox_inches="tight"
            )
        plt.close()

    def plot_activation_distributions(self, save_path: Optional[str] = None):
        # Matplotlib vizuals for activation statistics and dead neuron counts
        if not self.activation_histories:
            return

        _, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax_ac_avgs = axes[0, 0]
        for name, history in list(self.activation_histories.items())[:5]:
            steps = [h["step"] for h in history]
            means = [h["mean"] for h in history]
            ax_ac_avgs.plot(steps, means, label=name[:25], alpha=0.7)
        ax_ac_avgs.set_xlabel("Step")
        ax_ac_avgs.set_ylabel("Activation Mean")
        ax_ac_avgs.set_title("Activation Means")
        ax_ac_avgs.legend(fontsize=8)
        ax_ac_avgs.grid(True, alpha=0.3)

        ax_stddevs = axes[0, 1]
        for name, history in list(self.activation_histories.items())[:5]:
            steps = [h["step"] for h in history]
            stds = [h["std"] for h in history]
            ax_stddevs.plot(steps, stds, label=name[:25], alpha=0.7)
        ax_stddevs.set_xlabel("Step")
        ax_stddevs.set_ylabel("Activation Std")
        ax_stddevs.set_title("Activation Standard Deviations")
        ax_stddevs.legend(fontsize=8)
        ax_stddevs.grid(True, alpha=0.3)

        ax_ded_neur = axes[1, 0]
        for name, counts in self.dead_neurons.items():
            steps = list(range(len(counts)))
            ax_ded_neur.plot(steps, counts, label=name[:25], alpha=0.7)
        ax_ded_neur.set_xlabel("Step")
        ax_ded_neur.set_ylabel("Dead Neuron Count")
        ax_ded_neur.set_title("Dead Neurons Over Time")
        ax_ded_neur.legend(fontsize=8)
        ax_ded_neur.grid(True, alpha=0.3)

        ax_gradflow = axes[1, 1]
        flow_score = self.compute_gradient_flow_score()
        ax_gradflow.bar(
            ["Gradient Flow Score"],
            [flow_score],
            color=["green" if flow_score > 0.5 else "red"],
        )
        ax_gradflow.set_ylabel("Score")
        ax_gradflow.set_title("Gradient Flow Health")
        ax_gradflow.set_ylim(0, 1)
        ax_gradflow.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.savefig(
                self.log_dir / "activation_distributions.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close()

    def plot_weight_distributions(self, save_path: Optional[str] = None):
        # Plot weight norm evolution across training steps
        if not self.weight_histories:
            return

        _, ax = plt.subplots(figsize=(10, 6))

        for name, history in list(self.weight_histories.items())[:5]:
            steps, norms = [h["step"] for h in history], [h["norm"] for h in history]
            ax.plot(steps, norms, label=name[:30], alpha=0.7)

        ax.set_xlabel("Step")
        ax.set_ylabel("Weight Norm")
        ax.set_title("Weight Norms Over Time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.savefig(
                self.log_dir / "weight_distributions.png", dpi=150, bbox_inches="tight"
            )
        plt.close()

    def save_report(self, step: int):
        report = {
            "step": step,
            "gradient_flow_score": self.compute_gradient_flow_score(),
            "stability_check": self.check_numerical_stability(),
            "current_gradients": self.track_gradients(step),
            "current_weights": self.track_weights(step),
        }

        with open(self.log_dir / f"diagnostics_step_{step}.json", "w") as f:
            json.dump(report, f, indent=2)

        return report

    def save_all_histories(self):
        histories = {
            "gradients": self.gradient_histories,
            "activations": self.activation_histories,
            "weights": self.weight_histories,
            "dead_neurons": self.dead_neurons,
        }

        with open(self.log_dir / "all_histories.json", "w") as f:
            json.dump(histories, f)

    def generate_visualizations(self, step: int):
        self.plot_gradient_flow()
        self.plot_activation_distributions()
        self.plot_weight_distributions()
        self.save_report(step)
        self.save_all_histories()
