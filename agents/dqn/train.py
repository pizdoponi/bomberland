import asyncio
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from dqn_config import DQNConfig
from dqn_model import NUM_ACTIONS, ActionType, DQNModel, ReplayBuffer
from dqn_shared import DQNFeatureBuilder
from game_state import GameState
from torch.optim.adamw import AdamW
from types_ import MAX_CONCURRENT_BOMBS_PER_AGENT, SkipAction
from types_ import GameState as TypedGameState

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="[dqn-train] %(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set debug level from environment
if os.environ.get("DEBUG", "0") == "1":
    logger.setLevel(logging.DEBUG)


TRAINING_MODE_ENABLED = str(os.environ.get("TRAINING_MODE_ENABLED", "0")) == "1"
logger.info(f"{TRAINING_MODE_ENABLED=}")

agent_uri = (
    os.environ.get("GAME_CONNECTION_STRING")
    or "ws://game-engine:3000/?role=agent&agentId=agentA&name=dqn-train"
)
admin_uri = "ws://game-engine:3000/?role=admin"

logger.info(f"{admin_uri=}")
logger.info(f"{agent_uri=}")


@dataclass
class TrainingMetrics:
    """Track training metrics for monitoring progress."""

    # Per-game metrics
    game_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    game_lengths: Deque[int] = field(default_factory=lambda: deque(maxlen=100))
    game_wins: Deque[int] = field(default_factory=lambda: deque(maxlen=100))  # 1=win, 0=loss, 0.5=draw

    # Per-step metrics
    losses: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    q_values: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    td_errors: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    # Current game tracking
    current_game_reward: float = 0.0
    current_game_length: int = 0

    def reset_game(self):
        self.current_game_reward = 0.0
        self.current_game_length = 0

    def add_step_reward(self, reward: float):
        self.current_game_reward += reward
        self.current_game_length += 1

    def end_game(self, win: Optional[bool]):
        self.game_rewards.append(self.current_game_reward)
        self.game_lengths.append(self.current_game_length)
        if win is None:
            self.game_wins.append(0.5)
        else:
            self.game_wins.append(1.0 if win else 0.0)
        self.reset_game()

    def add_loss(self, loss: float):
        self.losses.append(loss)

    def add_q_value(self, q_val: float):
        self.q_values.append(q_val)

    def add_td_error(self, td_error: float):
        self.td_errors.append(td_error)

    def get_summary(self) -> Dict[str, float]:
        return {
            "avg_reward_100": np.mean(self.game_rewards) if self.game_rewards else 0.0,
            "avg_length_100": np.mean(self.game_lengths) if self.game_lengths else 0.0,
            "win_rate_100": np.mean(self.game_wins) if self.game_wins else 0.0,
            "avg_loss_1000": np.mean(self.losses) if self.losses else 0.0,
            "avg_q_1000": np.mean(self.q_values) if self.q_values else 0.0,
            "avg_td_error_1000": np.mean(self.td_errors) if self.td_errors else 0.0,
        }


class DQNTrainer:
    def __init__(self) -> None:
        self.admin_client = GameState(admin_uri)
        self.agent_client = GameState(agent_uri)

        self.config = DQNConfig()
        logger.info(f"config={self.config}")
        self._epsilon = self.config.epsilon_start

        self._feature_builder = DQNFeatureBuilder(self.config)
        self._replay_buffer = ReplayBuffer(self.config.replay_capacity)

        self._last_state: Dict[str, np.ndarray] = {}
        self._last_action: Dict[str, ActionType] = {}
        self._last_legal_actions: Dict[str, List[int]] = {}
        self._prev_game_state: Optional[TypedGameState] = None
        self._last_endgame_game_id: Optional[str] = None

        self._step_count = 0
        self._first_tick_event = asyncio.Event()

        self._games_played = 0
        self._training_start_time = time.time()

        # Metrics tracking
        self._metrics = TrainingMetrics()

        in_channels = self._feature_builder.num_channels * self.config.frame_stack_size
        self._model = DQNModel(
            conv_in_channels=in_channels,
            conv_hidden_channels=self.config.conv_hidden_channels,
            conv_out_channels=self.config.conv_out_channels,
            height=15,
            width=15,
            num_heads=self._feature_builder.num_heads,
            num_actions=NUM_ACTIONS,
            fc_hidden_dim=self.config.fc_hidden_dim,
        ).to(self.config.device)

        self._target_model = DQNModel(
            conv_in_channels=in_channels,
            conv_hidden_channels=self.config.conv_hidden_channels,
            conv_out_channels=self.config.conv_out_channels,
            height=15,
            width=15,
            num_heads=self._feature_builder.num_heads,
            num_actions=NUM_ACTIONS,
            fc_hidden_dim=self.config.fc_hidden_dim,
        ).to(self.config.device)

        self._optimizer = AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5,  # Small L2 regularization
        )

        # Learning rate scheduler - reduce LR when stuck
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode='max',
            factor=0.5,
            patience=50,  # In terms of log intervals
            verbose=True,
        )

        has_checkpoint = os.path.exists(self.config.load_path)
        if has_checkpoint:
            self._load_checkpoint()
        else:
            logger.info(
                f"No checkpoint found at {self.config.load_path}, training from scratch"
            )

        # Always start with a fresh target equal to the online network.
        self._target_model.load_state_dict(self._model.state_dict())
        self._target_model.eval()  # Target never trains

        # Log model info
        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(f"Model has {total_params:,} parameters")

    def _load_checkpoint(self):
        """Load model checkpoint with error handling."""
        try:
            checkpoint = torch.load(
                self.config.load_path,
                map_location=self.config.device,
                weights_only=True,
            )
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'step_count' in checkpoint:
                    self._step_count = checkpoint['step_count']
                if 'epsilon' in checkpoint:
                    self._epsilon = checkpoint['epsilon']
                if 'games_played' in checkpoint:
                    self._games_played = checkpoint['games_played']
                logger.info(f"Loaded full checkpoint from {self.config.load_path}")
                logger.info(f"  Resuming from step {self._step_count}, epsilon {self._epsilon:.4f}")
            else:
                # Old format - just state dict
                self._model.load_state_dict(checkpoint)
                logger.info(f"Loaded weights-only checkpoint from {self.config.load_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    def _save_checkpoint(self):
        """Save model checkpoint with training state."""
        os.makedirs(os.path.dirname(self.config.checkpoint_path) or ".", exist_ok=True)
        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'step_count': self._step_count,
            'epsilon': self._epsilon,
            'games_played': self._games_played,
            'metrics': self._metrics.get_summary(),
        }
        torch.save(checkpoint, self.config.checkpoint_path)
        logger.info(f"Saved checkpoint at step {self._step_count}")

    async def run(self):
        logger.info(f"Connection agent to game engine at {agent_uri}")
        agent_connection = await self.agent_client.connect()
        logger.info(f"Connection admin to game engine at {admin_uri}")
        admin_connection = await self.admin_client.connect()

        self.agent_client.set_game_tick_callback(self._on_game_tick)
        self.admin_client.set_endgame_callback(self._on_endgame)

        admin_task = asyncio.create_task(
            self.admin_client._handle_messages(admin_connection)  # type: ignore
        )
        agent_task = asyncio.create_task(
            self.agent_client._handle_messages(agent_connection)  # type: ignore
        )

        kickoff_task = asyncio.create_task(self._ensure_first_tick())
        logger.info("Creating admin, agent, and kickoff tasks")
        await asyncio.gather(admin_task, agent_task, kickoff_task)

    async def _on_game_tick(self, tick_number: int, game_state_: Dict):
        if not self._first_tick_event.is_set():
            logger.info(
                f"========== Game {self._games_played + 1} =========="
            )
            self._first_tick_event.set()

        self._step_count += 1

        game_state = TypedGameState.from_dict(game_state_)

        my_units_sorted = sorted([unit.unit_id for unit in game_state.my_units])

        frame = self._feature_builder.encode_frame(game_state)
        stacked_state = self._feature_builder.update_frame_stack(frame)

        if self._prev_game_state is not None:
            team_reward, units_reward, is_episode_done = (
                self._feature_builder.compute_team_and_unit_rewards(
                    self._prev_game_state, game_state, self._last_action
                )
            )
            self._metrics.add_step_reward(team_reward)
        else:
            team_reward, units_reward, is_episode_done = 0.0, {}, False

        timeout_reached = game_state.tick >= game_state.config.game_duration_ticks - 1
        episode_done = is_episode_done or timeout_reached

        if TRAINING_MODE_ENABLED and self._prev_game_state is not None:
            for unit_id, previous_state in list(self._last_state.items()):
                head_index = self._feature_builder.unit_id_to_head_index(
                    unit_id, my_units_sorted
                )
                if head_index is None:
                    self._last_state.pop(unit_id, None)
                    self._last_action.pop(unit_id, None)
                    self._last_legal_actions.pop(unit_id, None)
                    continue

                last_action = self._last_action.get(unit_id)
                if last_action is None:
                    self._last_state.pop(unit_id, None)
                    self._last_action.pop(unit_id, None)
                    self._last_legal_actions.pop(unit_id, None)
                    continue

                last_legal_actions = self._last_legal_actions.get(unit_id)
                if last_legal_actions is None:
                    self._last_state.pop(unit_id, None)
                    self._last_action.pop(unit_id, None)
                    self._last_legal_actions.pop(unit_id, None)
                    continue

                reward = units_reward.get(unit_id, team_reward)

                next_unit_state = game_state.get_unit(unit_id)
                legal_actions_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
                legal_actions_mask[ActionType.NOOP.value] = 1.0  # always allow NOOP
                unit_is_alive = (
                    next_unit_state is not None and next_unit_state.is_alive()
                )
                if unit_is_alive:
                    for action in last_legal_actions:
                        legal_actions_mask[action] = 1.0

                transition_done = 1.0 if (episode_done or not unit_is_alive) else 0.0

                self._replay_buffer.add(
                    previous_state,
                    head_index,
                    last_action,
                    reward,
                    stacked_state,
                    legal_actions_mask,
                    transition_done,
                )

                if not unit_is_alive:
                    self._last_state.pop(unit_id, None)
                    self._last_action.pop(unit_id, None)

        # Get Q-values for action selection
        state_tensor = (
            torch.from_numpy(stacked_state).float().unsqueeze(0).to(self.config.device)
        )
        with torch.no_grad():
            q_values = self._model(state_tensor)[0].cpu().numpy()
            # Track Q-values for monitoring
            self._metrics.add_q_value(float(np.mean(q_values)))

        agent_bombs_in_play = len(game_state.my_units_bombs())
        pending_bomb_placements = 0

        for unit_id in my_units_sorted:
            unit_state = game_state.get_unit(unit_id)
            if unit_state is None or not unit_state.is_alive():
                continue

            head_index = self._feature_builder.unit_id_to_head_index(
                unit_id, my_units_sorted
            )
            if head_index is None:
                continue

            legal_action_types = game_state.legal_actions(unit_state)
            if (
                ActionType.PLACE_BOMB in legal_action_types
                and agent_bombs_in_play + pending_bomb_placements
                >= MAX_CONCURRENT_BOMBS_PER_AGENT
            ):
                legal_action_types = [
                    action
                    for action in legal_action_types
                    if action != ActionType.PLACE_BOMB
                ]
            legal_actions = [action.value for action in legal_action_types]

            action_index = self._select_action(q_values[head_index], legal_actions)
            action_type = ActionType.from_index(action_index)

            if action_type == ActionType.PLACE_BOMB:
                pending_bomb_placements += 1

            if TRAINING_MODE_ENABLED and not episode_done:
                self._last_state[unit_id] = stacked_state
                self._last_action[unit_id] = action_type
                self._last_legal_actions[unit_id] = legal_actions

            logger.debug(f"Unit {unit_id} executing action {action_type}")
            await self._execute_action(unit_id, action_type, game_state)

        if TRAINING_MODE_ENABLED:
            self._train_step()

            if self._step_count % self.config.target_update_interval == 0:
                logger.info(f"Updating target network at step {self._step_count}")
                self._target_model.load_state_dict(self._model.state_dict())

            if self._step_count % self.config.save_interval == 0:
                self._save_checkpoint()

            # update epsilon
            if (
                self._epsilon > self.config.epsilon_min
                and len(self._replay_buffer) > self.config.warmup_steps
            ):
                self._epsilon = max(
                    self.config.epsilon_min,
                    self._epsilon * self.config.epsilon_decay,
                )

        self._prev_game_state = game_state

        # should happen in _on_endgame, but just in case
        if episode_done:
            self._last_state.clear()
            self._last_action.clear()

        await self.admin_client.send_request_tick()

    def _train_step(self) -> None:
        if (
            # not enough samples in replay buffer
            len(self._replay_buffer) < self.config.batch_size
            # not warmed up yet
            or len(self._replay_buffer) < self.config.warmup_steps
        ):
            return

        batch = self._replay_buffer.sample(self.config.batch_size)

        state_batch = torch.from_numpy(batch.states).float().to(self.config.device)
        next_state_batch = (
            torch.from_numpy(batch.next_states).float().to(self.config.device)
        )
        head_index_batch = (
            torch.from_numpy(batch.head_indices).long().to(self.config.device)
        )
        action_batch = torch.from_numpy(batch.actions).long().to(self.config.device)
        reward_batch = torch.from_numpy(batch.rewards).float().to(self.config.device)
        next_legal_action_mask_batch = (
            torch.from_numpy(batch.next_legal_actions_mask)
            .float()
            .to(self.config.device)
        )
        done_batch = torch.from_numpy(batch.dones).float().to(self.config.device)

        batch_indices = torch.arange(self.config.batch_size, device=self.config.device)

        q_values = self._model(state_batch)
        predicted_q = q_values[batch_indices, head_index_batch, action_batch]

        with torch.no_grad():
            # Double DQN: use online network to select actions, target to evaluate
            q_next_online = self._model(next_state_batch)
            q_next_online_head = q_next_online[batch_indices, head_index_batch]
            masked_q_next_online = q_next_online_head.masked_fill(
                next_legal_action_mask_batch == 0, -1e9
            )
            next_actions = torch.argmax(masked_q_next_online, dim=1)

            q_next_target = self._target_model(next_state_batch)
            q_next_target_head = q_next_target[batch_indices, head_index_batch]
            next_q_values = q_next_target_head.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            target_values = (
                reward_batch + (1.0 - done_batch) * self.config.gamma * next_q_values
            )

        # Huber loss for stability
        loss = F.smooth_l1_loss(predicted_q, target_values)

        # Track TD error for metrics
        with torch.no_grad():
            td_error = torch.abs(predicted_q - target_values).mean().item()
            self._metrics.add_td_error(td_error)

        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10.0)
        self._optimizer.step()

        self._metrics.add_loss(loss.item())

        if self._step_count % self.config.log_interval == 0:
            self._log_progress()

    def _log_progress(self):
        """Log training progress with comprehensive metrics."""
        elapsed = time.time() - self._training_start_time
        steps_per_sec = self._step_count / elapsed if elapsed > 0 else 0

        metrics = self._metrics.get_summary()

        logger.info(
            f"Step {self._step_count:,} | "
            f"Games {self._games_played} | "
            f"ε {self._epsilon:.4f} | "
            f"Loss {metrics['avg_loss_1000']:.6f} | "
            f"Q {metrics['avg_q_1000']:.3f} | "
            f"TD {metrics['avg_td_error_1000']:.4f} | "
            f"WinRate {metrics['win_rate_100']:.1%} | "
            f"AvgReward {metrics['avg_reward_100']:.3f} | "
            f"AvgLen {metrics['avg_length_100']:.0f} | "
            f"Buffer {len(self._replay_buffer):,} | "
            f"Steps/s {steps_per_sec:.1f}"
        )

        # Update LR scheduler based on win rate
        if self._games_played >= 100:
            self._scheduler.step(metrics['win_rate_100'])

    def _select_action(self, q_values: np.ndarray, legal_actions: List[int]) -> int:
        """Select action using epsilon-greedy with legal action masking."""
        if not legal_actions:
            return ActionType.NOOP.value

        if random.random() < self._epsilon:
            return random.choice(legal_actions)
        return max(legal_actions, key=lambda idx: q_values[idx])

    async def _execute_action(
        self, unit_id: str, action_type: ActionType, game_state: TypedGameState
    ) -> None:
        action_packet = action_type.to_action_packet(unit_id, game_state)
        if isinstance(action_packet, SkipAction):
            return
        await self.agent_client._send(action_packet.to_dict())

    async def _on_endgame(self, *args, **kwargs) -> None:
        # ignore duplicate endgame events sent by the game engine
        payload = args[0] if args else {}
        game_id = payload.get("game_id")
        if game_id is None:
            game_id = payload.get("initial_state", {}).get("game_id")
        if game_id is not None and game_id == self._last_endgame_game_id:
            return

        self._games_played += 1

        # Determine win/loss/draw
        win = None
        if self._prev_game_state is not None:
            my_alive = len(self._prev_game_state.my_alive_units)
            enemy_alive = len(self._prev_game_state.enemy_alive_units)
            if my_alive > enemy_alive:
                win = True
            elif enemy_alive > my_alive:
                win = False
            # else: draw, win stays None

        # Capture metrics before end_game() resets them
        game_length = self._metrics.current_game_length
        game_reward = self._metrics.current_game_reward

        self._metrics.end_game(win)

        win_str = "WIN" if win is True else ("LOSS" if win is False else "DRAW")
        logger.info(
            f"Game {self._games_played} ended: {win_str} | "
            f"Length: {game_length} | "
            f"Reward: {game_reward:.3f}"
        )

        self._feature_builder.reset_stack()
        self._last_state.clear()
        self._last_action.clear()
        self._prev_game_state = None
        if game_id is not None:
            self._last_endgame_game_id = game_id

        # Check if we've reached max steps
        if self._step_count >= self.config.max_steps:
            logger.info(f"Reached max steps ({self.config.max_steps}). Stopping training.")
            self._save_checkpoint()
            raise SystemExit(0)

        world_seed = random.randint(0, 2**31 - 1)
        await self.admin_client.send_request_game_reset(world_seed=world_seed)
        self._first_tick_event.clear()

    async def _ensure_first_tick(self) -> None:
        while not self._first_tick_event.is_set():
            await self.admin_client.send_request_tick()
            await asyncio.sleep(1)


def main() -> None:
    logger.info("=" * 60)
    logger.info("DQN Training Starting")
    logger.info("=" * 60)

    retries = 0
    max_retries = 10

    while retries < max_retries:
        try:
            trainer = DQNTrainer()
            asyncio.run(trainer.run())
            break
        except SystemExit:
            logger.info("Training completed successfully.")
            break
        except Exception as exc:
            retries += 1
            logger.error(f"Trainer error (attempt {retries}/{max_retries}): {exc}")
            if retries < max_retries:
                time.sleep(5)
            else:
                logger.error("Max retries reached. Exiting.")
                raise


if __name__ == "__main__":
    main()
