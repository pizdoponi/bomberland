from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DQNConfig:
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.99999
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 20000
    learning_rate: float = 0.00005
    fc_hidden_dim: int = 128
    conv_hidden_channels: int = 64
    conv_out_channels: int = 64
    target_update_interval: int = 200
    save_interval: int = 500
    checkpoint_path: str = "checkpoints/dqn_cnn_weights.pt"
    load_path: str = "checkpoints/dqn_cnn_weights.pt"
    frame_stack_size: int = 5
    max_unit_hp: float = 3.0
    max_bombs: float = 3.0
    max_ore_hp: float = 3.0
    min_bomb_armed_duration: float = 5.0
    max_bomb_armed_duration: float = 30.0
    max_blast_duration: float = 5.0
    max_powerup_duration: float = 40.0
    max_stunned_duration: float = 15.0
    max_invulnerable_duration: float = 10.0
    device: str = "cuda"
