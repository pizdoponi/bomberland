from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from types_ import ActionType

ACTION_ORDER = ActionType.ordered()
NUM_ACTIONS = len(ACTION_ORDER)


@dataclass
class Transition:
    state: np.ndarray
    head_index: int
    action: ActionType
    reward: float
    next_state: np.ndarray
    next_legal_actions_mask: np.ndarray
    done: float


class ReplayBufferSample(NamedTuple):
    states: np.ndarray
    head_indices: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    next_legal_actions_mask: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

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
        self._buffer.append(
            Transition(
                state=state,
                head_index=head_index,
                action=action,
                reward=reward,
                next_state=next_state,
                next_legal_actions_mask=next_legal_actions_mask,
                done=done,
            )
        )

    def sample(self, batch_size: int) -> ReplayBufferSample:
        batch = random.sample(self._buffer, batch_size)
        states = np.stack([t.state for t in batch])
        head_indices = np.array([t.head_index for t in batch], dtype=np.int64)
        actions = np.array([t.action.value for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        next_legal_actions_mask = np.stack(
            [t.next_legal_actions_mask for t in batch]
        ).astype(np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return ReplayBufferSample(
            states,
            head_indices,
            actions,
            rewards,
            next_states,
            next_legal_actions_mask,
            dones,
        )

    def __len__(self) -> int:
        return len(self._buffer)


class ResNetBlock(nn.Module):
    """
    Basic ResNet block (2x 3x3 conv)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

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
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.residual_connection(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

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
        self.stem = nn.Sequential(
            ResNetBlock(conv_in_channels, conv_hidden_channels),
            ResNetBlock(conv_hidden_channels, conv_hidden_channels),
            ResNetBlock(conv_hidden_channels, conv_hidden_channels),
            ResNetBlock(conv_hidden_channels, conv_out_channels),
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
        self.load_state_dict(
            torch.load(
                path, map_location="cuda" if torch.cuda.is_available() else "cpu"
            )
        )

    @staticmethod
    def from_checkpoint(path: str, **kwargs) -> "DQNModel":
        model = DQNModel(**kwargs)
        model.load(path)
        return model
