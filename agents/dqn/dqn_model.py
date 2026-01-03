from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from types_ import (
    ActionPacket,
    BombAction,
    DetonateAction,
    MoveAction,
    Point,
    SkipAction,
)


class ActionType(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PLACE_BOMB = 4
    DETONATE_BOMB = 5
    WAIT = 6
    SKIP = 6

    @classmethod
    def ordered(cls) -> list["ActionType"]:
        return [
            cls.UP,
            cls.DOWN,
            cls.LEFT,
            cls.RIGHT,
            cls.PLACE_BOMB,
            cls.DETONATE_BOMB,
            cls.WAIT,
        ]

    @classmethod
    def from_index(cls, index: int) -> "ActionType":
        return cls(index)

    def to_action_packet(
        self, unit_id: str, bomb_position: Optional[Point] = None
    ) -> ActionPacket:
        if self in {ActionType.DETONATE_BOMB}:
            assert bomb_position is not None, (
                "bomb_position must be provided for DETONATE_BOMB action"
            )

        mapping = {
            ActionType.UP: MoveAction.from_direction(unit_id=unit_id, direction="up"),
            ActionType.DOWN: MoveAction.from_direction(
                unit_id=unit_id, direction="down"
            ),
            ActionType.LEFT: MoveAction.from_direction(
                unit_id=unit_id, direction="left"
            ),
            ActionType.RIGHT: MoveAction.from_direction(
                unit_id=unit_id, direction="right"
            ),
            ActionType.PLACE_BOMB: BombAction(unit_id=unit_id),
            ActionType.DETONATE_BOMB: DetonateAction(
                unit_id=unit_id,
                target=bomb_position,  # pyright: ignore[reportArgumentType]
            ),
            ActionType.WAIT: SkipAction(unit_id=unit_id),
        }
        return mapping[self]


@dataclass
class Transition:
    state: np.ndarray
    head_index: int
    action: ActionType
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBufferSample(NamedTuple):
    states: np.ndarray
    head_indices: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
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
        done: float,
    ) -> None:
        self._buffer.append(
            Transition(
                state=state,
                head_index=head_index,
                action=action,
                reward=reward,
                next_state=next_state,
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
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return ReplayBufferSample(
            states, head_indices, actions, rewards, next_states, dones
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
        height: int,
        width: int,
        num_heads: int,
        num_actions: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._num_actions = num_actions

        # 7 convolutional layers, so that each tile can "see" the entire map
        self.trunk = nn.Sequential(
            nn.Conv2d(conv_in_channels, conv_hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                conv_hidden_channels, conv_hidden_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_hidden_channels, conv_hidden_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_hidden_channels, conv_hidden_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_hidden_channels, conv_hidden_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_hidden_channels, conv_hidden_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_hidden_channels, conv_out_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

        conv_out_dim = conv_out_channels * height * width
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_actions) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        features = self.fc(features)
        head_outputs = [head(features) for head in self.heads]
        return torch.stack(head_outputs, dim=1)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))


ACTION_ORDER = ActionType.ordered()
NUM_ACTIONS = len(ACTION_ORDER)
