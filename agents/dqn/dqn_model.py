from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from types_ import ActionType

ACTION_ORDER = ActionType.ordered()
NUM_ACTIONS = len(ACTION_ORDER)


class ReplayBufferSample(NamedTuple):
    """Batch of transitions, already on GPU as tensors."""
    states: torch.Tensor
    head_indices: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    next_legal_actions_mask: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Circular replay buffer stored entirely on GPU.

    Eliminates CPU→GPU transfer bottleneck during sampling by keeping all data
    on device. Uses float16 for states to reduce VRAM usage (~3.2GB for 50k capacity).
    """
    def __init__(
        self,
        capacity: int,
        state_shape: tuple = (72, 15, 15),
        num_actions: int = NUM_ACTIONS,
        device: str = "cuda",
    ):
        self._capacity = capacity
        self._state_shape = state_shape
        self._num_actions = num_actions
        self._device = torch.device(device)
        self._position = 0
        self._size = 0

        # Pre-allocate GPU tensors
        # Use float16 for states to reduce VRAM (~3.2GB vs ~6.5GB for 50k capacity)
        self._states = torch.zeros(
            (capacity, *state_shape), dtype=torch.float16, device=self._device
        )
        self._next_states = torch.zeros(
            (capacity, *state_shape), dtype=torch.float16, device=self._device
        )
        self._head_indices = torch.zeros(
            capacity, dtype=torch.int64, device=self._device
        )
        self._actions = torch.zeros(
            capacity, dtype=torch.int64, device=self._device
        )
        self._rewards = torch.zeros(
            capacity, dtype=torch.float32, device=self._device
        )
        self._next_legal_actions_mask = torch.zeros(
            (capacity, num_actions), dtype=torch.float32, device=self._device
        )
        self._dones = torch.zeros(
            capacity, dtype=torch.float32, device=self._device
        )

    def add(
        self,
        state: np.ndarray,
        head_index: int,
        action: ActionType,
        reward: float,
        next_state: np.ndarray,
        next_legal_actions_mask: np.ndarray,
        done: float,
    ) -> None:
        idx = self._position
        # Transfer single transition to GPU (negligible overhead)
        self._states[idx] = torch.from_numpy(state).to(dtype=torch.float16, device=self._device)
        self._next_states[idx] = torch.from_numpy(next_state).to(dtype=torch.float16, device=self._device)
        self._head_indices[idx] = head_index
        self._actions[idx] = action.value
        self._rewards[idx] = reward
        self._next_legal_actions_mask[idx] = torch.from_numpy(next_legal_actions_mask).to(device=self._device)
        self._dones[idx] = done

        self._position = (self._position + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> ReplayBufferSample:
        # Generate random indices on GPU
        indices = torch.randint(0, self._size, (batch_size,), device=self._device)

        # Direct GPU tensor indexing - no CPU↔GPU transfer, no numpy conversion
        # Convert states to float32 for training
        return ReplayBufferSample(
            states=self._states[indices].float(),
            head_indices=self._head_indices[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_states=self._next_states[indices].float(),
            next_legal_actions_mask=self._next_legal_actions_mask[indices],
            dones=self._dones[indices],
        )

    def __len__(self) -> int:
        return self._size


class ResNetBlock(nn.Module):
    """
    Basic ResNet block (2x 3x3 conv) using LayerNorm instead of BatchNorm.
    LayerNorm is more stable for RL because it doesn't track running statistics
    and works consistently regardless of train/eval mode switching.
    """

    def __init__(self, in_channels, out_channels, height=15, width=15, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        # LayerNorm over [C, H, W] - more stable for RL than BatchNorm
        self.ln1 = nn.LayerNorm([out_channels, height, width])

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.ln2 = nn.LayerNorm([out_channels, height, width])

        # residual connection (identity or projection)
        self.residual_connection = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.residual_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.LayerNorm([out_channels, height, width]),
            )

    def forward(self, x):
        identity = self.residual_connection(x)

        out = self.conv1(x)
        out = self.ln1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.ln2(out)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class DQNModel(nn.Module):
    def __init__(
        self,
        conv_in_channels: int,
        conv_hidden_channels: int,
        conv_out_channels: int,
        fc_hidden_dim: int,
        height: int = 15,
        width: int = 15,
        num_heads: int = 3,  # 3 units per agent
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        # > 7 convolutional layers, so that each tile can "see" the entire map
        # Pass height/width to ResNetBlock for LayerNorm dimensions
        self.stem = nn.Sequential(
            ResNetBlock(conv_in_channels, conv_hidden_channels, height, width),
            ResNetBlock(conv_hidden_channels, conv_hidden_channels, height, width),
            ResNetBlock(conv_hidden_channels, conv_hidden_channels, height, width),
            ResNetBlock(conv_hidden_channels, conv_out_channels, height, width),
        )

        conv_out_dim = conv_out_channels * height * width
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, fc_hidden_dim),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList(
            [nn.Linear(fc_hidden_dim, num_actions) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.fc(out)
        head_outputs = [head(out) for head in self.heads]
        return torch.stack(
            head_outputs, dim=1
        )  # shape: (batch_size, num_heads, num_actions)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(
            path,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            weights_only=True,
        )
        # Handle both new checkpoint format and legacy format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)

    @staticmethod
    def from_checkpoint(path: str, **kwargs) -> "DQNModel":
        model = DQNModel(**kwargs)
        model.load(path)
        return model
