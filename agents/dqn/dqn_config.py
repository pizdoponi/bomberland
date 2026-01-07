from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DQNConfig:
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05  # Lower minimum for more exploitation
    epsilon_decay: float = 0.99995  # Slower decay for more exploration early on

    # Learning parameters
    gamma: float = 0.99
    batch_size: int = 256  # Larger batches for more stable gradients
    replay_capacity: int = 200_000  # Much larger buffer for diverse experience
    warmup_steps: int = 10_000  # More warmup to fill buffer before learning
    learning_rate: float = 0.0003  # Slightly higher LR with Adam

    # Network architecture
    fc_hidden_dim: int = 256  # Larger hidden layer
    conv_hidden_channels: int = 64
    conv_out_channels: int = 64

    # Training intervals
    target_update_interval: int = 2000  # Less frequent target updates for stability
    save_interval: int = 5000  # Save less frequently
    log_interval: int = 500  # Log every 500 steps

    # Checkpointing
    checkpoint_path: str = "checkpoints/dqn_cnn_weights.pt"
    load_path: str = "checkpoints/dqn_cnn_weights.pt"

    # Feature encoding
    frame_stack_size: int = 4  # 4 frames is often enough

    # Normalization constants
    max_unit_hp: float = 3.0
    max_bombs: float = 3.0
    max_ore_hp: float = 3.0
    min_bomb_armed_duration: float = 5.0
    max_bomb_armed_duration: float = 30.0
    max_blast_duration: float = 5.0
    max_powerup_duration: float = 40.0
    max_stunned_duration: float = 15.0
    max_invulnerable_duration: float = 10.0

    # Device
    device: str = "cuda"

    # Training control
    max_steps: int = 500_000  # Target training steps
    eval_interval: int = 10_000  # Evaluate every N steps
    eval_games: int = 20  # Number of games for evaluation

    def __post_init__(self) -> None:
        """Fallback to CPU if CUDA is unavailable to avoid runtime crashes."""
        if "cuda" in self.device.lower() and not torch.cuda.is_available():
            self.device = "cpu"
