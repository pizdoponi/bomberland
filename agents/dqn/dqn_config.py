from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DQNConfig:
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.999990  # Math: 0.999990^300000 ≈ 0.05    

    # Learning parameters
    gamma: float = 0.99
    batch_size: int = 1024
    replay_capacity: int = 200_000
    warmup_steps: int = 5_000
    learning_rate: float = 0.0003  # Slightly conservative for stability
    train_every_n_steps: int = 4  # 1 gradient update per 4 env steps

    # Network architecture
    fc_hidden_dim: int = 256
    conv_hidden_channels: int = 64
    conv_out_channels: int = 64

    # Training intervals
    target_update_interval: int = 1000
    save_interval: int = 5000
    log_interval: int = 500

    # Checkpointing
    checkpoint_path: str = "checkpoints/dqn_cnn_weights.pt"
    load_path: str = "checkpoints/dqn_cnn_weights.pt"

    # Feature encoding
    frame_stack_size: int = 4

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
    max_steps: int = 500_000
    eval_interval: int = 10_000
    eval_games: int = 20

    def __post_init__(self) -> None:
        """Fallback to CPU if CUDA is unavailable to avoid runtime crashes."""
        if "cuda" in self.device.lower() and not torch.cuda.is_available():
            self.device = "cpu"
