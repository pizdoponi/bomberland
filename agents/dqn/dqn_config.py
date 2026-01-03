from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DQNConfig:
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.995
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 20000
    learning_rate: float = 0.0005
    hidden_dim: int = 128
    conv_hidden_channels: int = 64
    conv_out_channels: int = 64
    target_update_interval: int = 200
    save_interval: int = 500
    checkpoint_path: str = "checkpoints/dqn_cnn_weights.pt"
    load_path: str = "checkpoints/dqn_cnn_weights.pt"
    frame_stack_size: int = 4
    max_unit_hp: float = 3.0
    max_bombs: float = 3.0
    max_ore_hp: float = 3.0
    min_bomb_armed_duration: float = 5.0
    max_bomb_armed_duration: float = 30.0
    max_blast_duration: float = 5.0
    max_powerup_duration: float = 40.0
    max_stunned_duration: float = 15.0
    max_invulnerable_duration: float = 10.0
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "DQNConfig":
        checkpoint_path = os.environ.get("DQN_CHECKPOINT_PATH", cls.checkpoint_path)
        load_path = os.environ.get("DQN_LOAD_PATH", checkpoint_path)

        return cls(
            epsilon_start=float(os.environ.get("DQN_EPSILON_START", cls.epsilon_start)),
            epsilon_min=float(os.environ.get("DQN_EPSILON_MIN", cls.epsilon_min)),
            epsilon_decay=float(os.environ.get("DQN_EPSILON_DECAY", cls.epsilon_decay)),
            gamma=float(os.environ.get("DQN_GAMMA", cls.gamma)),
            batch_size=int(os.environ.get("DQN_BATCH_SIZE", cls.batch_size)),
            replay_capacity=int(
                os.environ.get("DQN_REPLAY_CAPACITY", cls.replay_capacity)
            ),
            learning_rate=float(os.environ.get("DQN_LEARNING_RATE", cls.learning_rate)),
            hidden_dim=int(os.environ.get("DQN_HIDDEN_DIM", cls.hidden_dim)),
            conv_hidden_channels=int(
                os.environ.get(
                    "DQN_CONV_HIDDEN_CHANNELS", cls.conv_hidden_channels
                )
            ),
            conv_out_channels=int(
                os.environ.get("DQN_CONV_OUT_CHANNELS", cls.conv_out_channels)
            ),
            target_update_interval=int(
                os.environ.get("DQN_TARGET_UPDATE_INTERVAL", cls.target_update_interval)
            ),
            save_interval=int(os.environ.get("DQN_SAVE_INTERVAL", cls.save_interval)),
            checkpoint_path=checkpoint_path,
            load_path=load_path,
            frame_stack_size=int(
                os.environ.get("DQN_FRAME_STACK", cls.frame_stack_size)
            ),
            max_unit_hp=float(os.environ.get("DQN_MAX_UNIT_HP", cls.max_unit_hp)),
            max_bombs=float(os.environ.get("DQN_MAX_BOMBS", cls.max_bombs)),
            max_ore_hp=float(os.environ.get("DQN_MAX_ORE_HP", cls.max_ore_hp)),
            device=os.environ.get("DQN_DEVICE", cls.device),
        )
