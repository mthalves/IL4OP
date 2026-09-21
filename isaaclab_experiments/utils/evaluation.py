"""Episode statistics for policy evaluation."""

import torch


class EpisodeStats:
    """Accumulates per-environment returns and lengths and summarises completed episodes."""

    def __init__(self, num_envs: int, device: str | torch.device):
        self.returns = torch.zeros(num_envs, device=device)
        self.lengths = torch.zeros(num_envs, device=device)
        self.episode_returns: list[float] = []
        self.episode_lengths: list[float] = []

    def update(self, rewards: torch.Tensor, dones: torch.Tensor):
        rewards = rewards.reshape(-1).to(self.returns.device)
        dones = dones.reshape(-1).to(self.returns.device).bool()
        self.returns += rewards
        self.lengths += 1
        if dones.any():
            self.episode_returns.extend(self.returns[dones].tolist())
            self.episode_lengths.extend(self.lengths[dones].tolist())
            self.returns[dones] = 0.0
            self.lengths[dones] = 0.0

    @property
    def num_episodes(self) -> int:
        return len(self.episode_returns)

    def summary(self) -> dict[str, float]:
        if not self.episode_returns:
            return {}
        returns = torch.tensor(self.episode_returns)
        lengths = torch.tensor(self.episode_lengths)
        return {
            "episodes": float(len(returns)),
            "return_mean": returns.mean().item(),
            "return_std": returns.std().item() if len(returns) > 1 else 0.0,
            "return_min": returns.min().item(),
            "return_max": returns.max().item(),
            "length_mean": lengths.mean().item(),
        }

    def report(self, total_steps: int):
        print("\n" + "=" * 60)
        print(f"Evaluation summary ({total_steps} steps, {self.returns.numel()} environments)")
        print("=" * 60)
        summary = self.summary()
        if not summary:
            print("No episode was completed; increase --max_steps to obtain statistics.")
        else:
            print(f"  completed episodes : {int(summary['episodes'])}")
            print(f"  return  mean ± std : {summary['return_mean']:.3f} ± {summary['return_std']:.3f}")
            print(f"  return  min / max  : {summary['return_min']:.3f} / {summary['return_max']:.3f}")
            print(f"  length  mean       : {summary['length_mean']:.1f} steps")
        print("=" * 60)
