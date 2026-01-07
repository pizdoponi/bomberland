import asyncio
import logging
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from dqn_config import DQNConfig
from dqn_model import NUM_ACTIONS, ActionType, DQNModel, ReplayBuffer
from dqn_shared import DQNFeatureBuilder
from game_state import GameState
from torch.optim.adamw import AdamW
from types_ import GameState as TypedGameState
from types_ import SkipAction

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="[dqn-train] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


TRAINING_MODE_ENABLED = str(os.environ.get("TRAINING_MODE_ENABLED", "0")) == "1"
logger.info(f"{TRAINING_MODE_ENABLED=}")

agent_uri = (
    os.environ.get("GAME_CONNECTION_STRING")
    or "ws://game-engine:3000/?role=agent&agentId=agentA&name=dqn-train"
)
admin_uri = "ws://game-engine:3000/?role=admin"

logger.info(f"{admin_uri=}")
logger.info(f"{agent_uri=}")


class DQNTrainer:
    def __init__(self) -> None:
        self.admin_client = GameState(admin_uri)
        self.agent_client = GameState(agent_uri)

        self.config = DQNConfig(epsilon_decay=0.99999, learning_rate=0.0001)
        logger.info(f"config={self.config}")
        self._epsilon = self.config.epsilon_start

        self._feature_builder = DQNFeatureBuilder(self.config)
        self._replay_buffer = ReplayBuffer(self.config.replay_capacity)

        self._last_state: Dict[str, np.ndarray] = {}
        self._last_action: Dict[str, ActionType] = {}
        self._last_legal_actions: Dict[str, List[int]] = {}
        self._prev_game_state: Optional[TypedGameState] = None

        self._step_count = 0
        self._first_tick_event = asyncio.Event()

        self._games_played = 0

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

        self._optimizer = AdamW(self._model.parameters(), lr=self.config.learning_rate)

        has_checkpoint = os.path.exists(self.config.load_path)
        if has_checkpoint:
            self._model.load(self.config.load_path)
            logger.info("Loaded checkpoint from %s", self.config.load_path)
        else:
            logger.info(
                f"No checkpoint found at {self.config.load_path}, training from scratch"
            )

        # Always start with a fresh target equal to the online network.
        self._target_model.load_state_dict(self._model.state_dict())

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
                f"--------------------- Game {self._games_played + 1} ----------------------"
            )
            logger.debug(
                f"First tick ({tick_number=}) received for game {self._games_played + 1}, setting first_tick_event"
            )
            self._first_tick_event.set()

        logger.debug(f"Step={self._step_count}, {tick_number=}")

        self._step_count += 1

        game_state = TypedGameState.from_dict(game_state_)

        my_units_sorted = sorted([unit.unit_id for unit in game_state.my_units])
        logger.debug(f"{my_units_sorted=}")

        frame = self._feature_builder.encode_frame(game_state)
        stacked_state = self._feature_builder.update_frame_stack(frame)

        logger.debug(
            f"frame.shape={frame.shape}, stacked_state.shape={stacked_state.shape}"
        )

        if self._prev_game_state is not None:
            team_reward, units_reward, is_episode_done = (
                self._feature_builder.compute_team_and_unit_rewards(
                    self._prev_game_state, game_state, self._last_action
                )
            )
            _units_rewards_str = (
                "{" + ", ".join(f"{k}: {v:.3f}" for k, v in units_reward.items()) + "}"
            )
            logger.debug(
                f"Step={self._step_count}, {team_reward=:.3f}, units_reward={_units_rewards_str}, {is_episode_done=}"
            )
        else:
            team_reward, units_reward, is_episode_done = 0.0, {}, False

        timeout_reached = game_state.tick >= game_state.config.game_duration_ticks - 1
        episode_done = is_episode_done or timeout_reached

        if TRAINING_MODE_ENABLED and self._prev_game_state is not None:
            for unit_id, previous_state in list(self._last_state.items()):
                head_index = self._feature_builder.unit_id_to_head_index(
                    unit_id, my_units_sorted
                )
                logger.debug(f"{unit_id=}, {head_index=}")
                if head_index is None:
                    logger.error(
                        f"Head index for unit {unit_id} is None, dropping cached transition"
                    )
                    self._last_state.pop(unit_id, None)
                    self._last_action.pop(unit_id, None)
                    self._last_legal_actions.pop(unit_id, None)
                    continue

                last_action = self._last_action.get(unit_id)
                if last_action is None:
                    logger.error(
                        f"Last action for unit {unit_id} missing, dropping cached transition"
                    )
                    self._last_state.pop(unit_id, None)
                    self._last_action.pop(unit_id, None)
                    self._last_legal_actions.pop(unit_id, None)
                    continue
                logger.debug(f"{unit_id=}, {last_action=}")

                last_legal_actions = self._last_legal_actions.get(unit_id)
                if last_legal_actions is None:
                    logger.error(
                        f"Last legal actions for unit {unit_id} missing, dropping cached transition"
                    )
                    self._last_state.pop(unit_id, None)
                    self._last_action.pop(unit_id, None)
                    self._last_legal_actions.pop(unit_id, None)
                    continue
                logger.debug(f"{unit_id=}, {last_legal_actions=}")

                reward = units_reward.get(unit_id, team_reward)

                if reward == team_reward:
                    my_alive_units = {
                        unit.unit_id for unit in game_state.my_alive_units
                    }
                    is_unit_alive = (
                        game_state.get_unit(unit_id) is not None
                        and game_state.get_unit(unit_id) in my_alive_units
                    )
                    logger.warning(
                        f"Unit {unit_id} has no individual reward, using team reward {team_reward}. Unit is alive: {is_unit_alive}."
                    )
                logger.debug(f"{unit_id=}, {reward=}")

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

        state_tensor = (
            torch.from_numpy(stacked_state).float().unsqueeze(0).to(self.config.device)
        )
        self._model.eval()
        with torch.no_grad():
            q_values = self._model(state_tensor)[0].cpu().numpy()
        self._model.train()

        for unit_id in my_units_sorted:
            unit_state = game_state.get_unit(unit_id)
            if unit_state is None or not unit_state.is_alive():
                continue

            head_index = self._feature_builder.unit_id_to_head_index(
                unit_id, my_units_sorted
            )
            logger.debug(f"{unit_id=}, {head_index=}")
            if head_index is None:
                continue

            legal_action_types = game_state.legal_actions(unit_state)
            legal_actions = [action.value for action in legal_action_types]

            action_index = self._select_action(q_values[head_index], legal_actions)
            action_type = ActionType.from_index(action_index)

            if TRAINING_MODE_ENABLED and not episode_done:
                self._last_state[unit_id] = stacked_state
                self._last_action[unit_id] = action_type
                self._last_legal_actions[unit_id] = legal_actions

            await self._execute_action(unit_id, action_type, game_state)

        if TRAINING_MODE_ENABLED:
            self._train_step()

            if self._step_count % self.config.target_update_interval == 0:
                logger.debug(f"Updating target network at step {self._step_count}")
                self._target_model.load_state_dict(self._model.state_dict())
            if self._step_count % self.config.save_interval == 0:
                logger.info(
                    f"Saving model checkpoint at step {self._step_count} to {self.config.checkpoint_path}"
                )
                self._model.save(self.config.checkpoint_path)

            # update epsilon
            if (
                self._epsilon > self.config.epsilon_min
                and len(self._replay_buffer) > self.config.warmup_steps
            ):
                new_epsilon = max(
                    self.config.epsilon_min,
                    self._epsilon * self.config.epsilon_decay,
                )
                logger.debug(
                    f"Updating epsilon from {self._epsilon} to {new_epsilon} at step {self._step_count}"
                )
                self._epsilon = new_epsilon

        logger.debug(
            "Updating previous game state with current game state for next tick"
        )
        self._prev_game_state = game_state

        # should happen in _on_endgame, but just in case
        if episode_done:
            self._last_state.clear()
            self._last_action.clear()

        logger.debug("Requesting next game tick")
        await self.admin_client.send_request_tick()

    def _train_step(self) -> None:
        if (
            # not enough samples in replay buffer
            len(self._replay_buffer) < self.config.batch_size
            # not warmed up yet
            or len(self._replay_buffer) < self.config.warmup_steps
        ):
            logger.debug(
                f"Skipping training step; len(replay_buffer)={len(self._replay_buffer)}, batch_size={self.config.batch_size}, warmup_steps={self.config.warmup_steps}"
            )
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

        loss = F.smooth_l1_loss(predicted_q, target_values)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()

        if self._step_count % 100 == 0:
            logger.info(
                f"Step={self._step_count}, loss={loss.item():.6f}, epsilon={self._epsilon:.3f}, replay_buffer_size={len(self._replay_buffer)}",
            )

    def _select_action(self, q_values: np.ndarray, legal_actions: List[int]) -> int:
        """Select action using epsilon-greedy with legal action masking."""
        if not legal_actions:
            logger.warning("No legal actions available, defaulting to NOOP")
            return ActionType.NOOP.value

        if (random_value := random.random()) < self._epsilon:
            logger.debug(
                f"Selecting random legal action due to {random_value=} < epsilon={self._epsilon:.3f} from {legal_actions=}",
            )
            return random.choice(legal_actions)
        logger.debug(f"Selecting greedy action from {legal_actions=}")
        return max(legal_actions, key=lambda idx: q_values[idx])

    async def _execute_action(
        self, unit_id: str, action_type: ActionType, game_state: TypedGameState
    ) -> None:
        logger.debug(f"Executing action {action_type} for unit {unit_id}")
        action_packet = action_type.to_action_packet(unit_id, game_state)
        if isinstance(action_packet, SkipAction):
            logger.debug(f"Skipping action for unit {unit_id}, because {action_type=}")
            return
        else:
            logger.debug(f"Sending action packet for {action_type=} for unit {unit_id}")
        await self.agent_client._send(action_packet.to_dict())

    async def _on_endgame(self, *args, **kwargs) -> None:
        self._games_played += 1
        logger.info(
            f"Step={self._step_count}, Game {self._games_played} lasted for {self._prev_game_state.tick if self._prev_game_state else 0} ticks."
        )
        logger.info(
            f"---------------------- End of Game {self._games_played} ----------------------"
        )
        self._feature_builder.reset_stack()
        self._last_state.clear()
        self._last_action.clear()
        self._prev_game_state = None
        world_seed = random.randint(0, 2**31 - 1)
        logger.info(f"Starting new game with world seed {world_seed}")
        await self.admin_client.send_request_game_reset(world_seed=world_seed)
        # reset _first_tick_event to wait for the next game start
        self._first_tick_event.clear()

    async def _ensure_first_tick(self) -> None:
        # Initial request_tick can be ignored if the game hasn't started yet.
        while not self._first_tick_event.is_set():
            await self.admin_client.send_request_tick()
            await asyncio.sleep(1)


def main() -> None:
    for _ in range(0, 10):
        while True:
            try:
                trainer = DQNTrainer()
                asyncio.run(trainer.run())

            except Exception as exc:
                logger.error("Trainer error: %s", exc)
                time.sleep(5)
                continue
            break


if __name__ == "__main__":
    main()
