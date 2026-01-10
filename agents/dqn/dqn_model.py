from __future__ import annotations

import os
from typing import NamedTuple, Optional

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


class PrioritizedReplayBufferSample(NamedTuple):
    """Batch of transitions with PER metadata."""
    states: torch.Tensor
    head_indices: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    next_legal_actions_mask: torch.Tensor
    dones: torch.Tensor
    indices: np.ndarray  # For updating priorities
    weights: torch.Tensor  # Importance sampling weights


class SumTree:
    """Binary tree where parent = sum of children.

    Enables O(log n) sampling proportional to priorities and O(log n) updates.
    Leaf nodes store priorities, internal nodes store sums.

    Tree structure for capacity=4:
              [sum_all]         <- index 0 (root)
             /         \
        [sum_01]     [sum_23]   <- indices 1, 2
        /    \       /    \
      [p0]  [p1]   [p2]  [p3]   <- indices 3,4,5,6 (leaves)

    Leaf indices: capacity-1 to 2*capacity-2
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        # Tree has 2*capacity - 1 nodes (capacity leaves + capacity-1 internal)
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def _leaf_to_tree_idx(self, leaf_idx: int) -> int:
        """Convert buffer index to tree index."""
        return leaf_idx + self._capacity - 1

    def _tree_to_leaf_idx(self, tree_idx: int) -> int:
        """Convert tree index to buffer index."""
        return tree_idx - self._capacity + 1

    def update(self, leaf_idx: int, priority: float) -> None:
        """Update priority of a leaf and propagate change up the tree."""
        tree_idx = self._leaf_to_tree_idx(leaf_idx)
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority

        # Propagate up to root
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += delta

    def sample(self, value: float) -> int:
        """Sample a leaf index given a value in [0, total_priority).

        Descends the tree: go left if value < left_child, else go right.
        """
        tree_idx = 0  # Start at root

        while True:
            left_idx = 2 * tree_idx + 1
            right_idx = left_idx + 1

            # Reached a leaf
            if left_idx >= len(self._tree):
                return self._tree_to_leaf_idx(tree_idx)

            if value <= self._tree[left_idx]:
                tree_idx = left_idx
            else:
                value -= self._tree[left_idx]
                tree_idx = right_idx

    def get(self, leaf_idx: int) -> float:
        """Get priority of a leaf."""
        return self._tree[self._leaf_to_tree_idx(leaf_idx)]

    @property
    def total(self) -> float:
        """Total sum of all priorities (root value)."""
        return self._tree[0]

    @property
    def max(self) -> float:
        """Maximum priority among leaves."""
        return np.max(self._tree[self._capacity - 1:])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with GPU storage.

    Samples transitions proportional to their TD error priority.
    Uses importance sampling weights to correct for non-uniform sampling bias.

    Key hyperparameters:
    - alpha: Controls prioritization (0 = uniform, 1 = full priority)
    - beta: Controls importance sampling correction (anneals 0.4 -> 1.0)
    - priority_epsilon: Small constant for numerical stability

    Reference: Schaul et al. "Prioritized Experience Replay" (2015)
    """

    def __init__(
        self,
        capacity: int,
        state_shape: tuple = (72, 15, 15),
        num_actions: int = NUM_ACTIONS,
        device: str = "cuda",
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 500_000,
        priority_epsilon: float = 1e-6,
    ):
        self._capacity = capacity
        self._state_shape = state_shape
        self._num_actions = num_actions
        self._device = torch.device(device)
        self._position = 0
        self._size = 0

        # PER parameters
        self._alpha = alpha
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._beta_steps = beta_steps
        self._priority_epsilon = priority_epsilon
        self._current_step = 0
        self._max_priority = 1.0  # Track max for new transitions

        # Sum tree for efficient priority sampling
        self._sum_tree = SumTree(capacity)

        # Pre-allocate GPU tensors (same as uniform buffer)
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

    @property
    def beta(self) -> float:
        """Current beta value (annealed from beta_start to beta_end)."""
        fraction = min(1.0, self._current_step / self._beta_steps)
        return self._beta_start + fraction * (self._beta_end - self._beta_start)

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
        """Add transition with maximum priority (will be corrected on first sample)."""
        idx = self._position

        # Store transition on GPU
        self._states[idx] = torch.from_numpy(state).to(dtype=torch.float16, device=self._device)
        self._next_states[idx] = torch.from_numpy(next_state).to(dtype=torch.float16, device=self._device)
        self._head_indices[idx] = head_index
        self._actions[idx] = action.value
        self._rewards[idx] = reward
        self._next_legal_actions_mask[idx] = torch.from_numpy(next_legal_actions_mask).to(device=self._device)
        self._dones[idx] = done

        # New transitions get max priority to ensure they're sampled at least once
        priority = self._max_priority ** self._alpha
        self._sum_tree.update(idx, priority)

        self._position = (self._position + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> PrioritizedReplayBufferSample:
        """Sample batch proportional to priorities with importance sampling weights."""
        self._current_step += 1

        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Divide total priority into batch_size segments for stratified sampling
        # This reduces variance compared to pure proportional sampling
        segment_size = self._sum_tree.total / batch_size

        for i in range(batch_size):
            # Sample uniformly within segment, then find corresponding leaf
            low = segment_size * i
            high = segment_size * (i + 1)
            value = np.random.uniform(low, high)

            idx = self._sum_tree.sample(value)
            # Ensure valid index (can happen with numerical issues)
            idx = min(idx, self._size - 1)

            indices[i] = idx
            priorities[i] = self._sum_tree.get(idx)

        # Compute importance sampling weights
        # w_i = (1/N * 1/P(i))^beta / max_w
        probs = priorities / self._sum_tree.total
        weights = (self._size * probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize by max for stability

        # Convert indices to tensor for GPU indexing
        indices_tensor = torch.from_numpy(indices).to(self._device)
        weights_tensor = torch.from_numpy(weights.astype(np.float32)).to(self._device)

        return PrioritizedReplayBufferSample(
            states=self._states[indices_tensor].float(),
            head_indices=self._head_indices[indices_tensor],
            actions=self._actions[indices_tensor],
            rewards=self._rewards[indices_tensor],
            next_states=self._next_states[indices_tensor].float(),
            next_legal_actions_mask=self._next_legal_actions_mask[indices_tensor],
            dones=self._dones[indices_tensor],
            indices=indices,
            weights=weights_tensor,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            # Priority = |TD_error| + epsilon (avoid zero priority)
            priority = (abs(td_error) + self._priority_epsilon) ** self._alpha
            self._sum_tree.update(idx, priority)
            self._max_priority = max(self._max_priority, abs(td_error) + self._priority_epsilon)

    def __len__(self) -> int:
        return self._size


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
