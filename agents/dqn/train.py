import asyncio
import logging
import os
import random
import time
from typing import Dict

import numpy as np
import torch
from dqn_config import DQNConfig
from dqn_model import NUM_ACTIONS, ActionType, DQNModel, ReplayBuffer
from dqn_shared import DQNFeatureBuilder
from game_state import GameState
from torch.optim.adamw import AdamW
from types_ import GameState as TypedGameState
from types_ import SkipAction

logging.basicConfig(
    level=logging.INFO,
    format="[dqn-train] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

agent_uri = (
    os.environ.get("GAME_CONNECTION_STRING")
    or "ws://game-engine:3000/?role=agent&agentId=agentA&name=dqn-train"
)
admin_uri = "ws://game-engine:3000/?role=admin"


class DQNTrainer:
    def __init__(self) -> None:
        self.admin_client = GameState(admin_uri)
        self.agent_client = GameState(agent_uri)

        self.config = DQNConfig()

        self._feature_builder = DQNFeatureBuilder(self.config)

        self._model: DQNModel = None  # pyright: ignore[reportAttributeAccessIssue]
        self._target_model: DQNModel = None  # pyright: ignore[reportAttributeAccessIssue]
        self._optimizer: AdamW = None  # pyright: ignore[reportAttributeAccessIssue]
        self._replay_buffer = ReplayBuffer(self.config.replay_capacity)

        self._last_state: Dict[str, np.ndarray] = {}
        self._last_action: Dict[str, ActionType] = {}
        self._last_metrics: Dict[str, Dict[str, float]] = {}
        self._step_count = 0
        self._first_tick_event = asyncio.Event()

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
            logger.info(f"First game tick received ({tick_number=}), starting training")
            self._first_tick_event.set()

        self._step_count += 1

        game_state = TypedGameState.from_dict(game_state_)

        my_units_sorted = sorted([unit.unit_id for unit in game_state.my_units])

        if self._model is None:
            in_channels = (
                self._feature_builder.num_channels * self.config.frame_stack_size
            )
            self._model = DQNModel(
                conv_in_channels=in_channels,
                conv_hidden_channels=self.config.conv_hidden_channels,
                conv_out_channels=self.config.conv_out_channels,
                height=game_state.world.height,
                width=game_state.world.width,
                num_heads=self._feature_builder.num_heads,
                num_actions=NUM_ACTIONS,
                fc_hidden_dim=self.config.fc_hidden_dim,
            ).to(self.config.device)
            self._target_model = DQNModel(
                conv_in_channels=in_channels,
                conv_hidden_channels=self.config.conv_hidden_channels,
                conv_out_channels=self.config.conv_out_channels,
                height=game_state.world.height,
                width=game_state.world.width,
                num_heads=self._feature_builder.num_heads,
                num_actions=NUM_ACTIONS,
                fc_hidden_dim=self.config.fc_hidden_dim,
            ).to(self.config.device)
            self._optimizer = AdamW(
                self._model.parameters(), lr=self.config.learning_rate
            )
            if os.path.exists(self.config.load_path):
                self._model.load(self.config.load_path)
                self._target_model.load_state_dict(self._model.state_dict())
                logger.info("Loaded checkpoint from %s", self.config.load_path)
            else:
                logger.info(
                    f"No checkpoint found at {self.config.load_path}, training from scratch"
                )

        frame = self._feature_builder.encode_frame(game_state)
        stacked_state = self._feature_builder.update_frame_stack(frame)

        num_enemy_units_alive = len(game_state.enemy_alive_units) > 0

        for unit_id in list(self._last_state.keys()):
            unit_state = game_state.get_unit(unit_id)
            unit_alive = unit_state.is_alive() if unit_state else False
            if not unit_alive:
                current_metrics = self._feature_builder.extract_metrics(
                    game_state, unit_id, game_state.enemy_units
                )
                reward = self._feature_builder.compute_reward(
                    self._last_metrics.get(unit_id, {}),
                    current_metrics,
                    False,
                    num_enemy_units_alive,
                )
                head_index = self._feature_builder.unit_id_to_head_index(
                    unit_id, my_units_sorted
                )
                if head_index is not None:
                    self._replay_buffer.add(
                        self._last_state[unit_id],
                        head_index,
                        self._last_action[unit_id],
                        reward,
                        self._last_state[unit_id],
                        1.0,
                    )
                self._last_state.pop(unit_id, None)
                self._last_action.pop(unit_id, None)
                self._last_metrics.pop(unit_id, None)

        state_tensor = (
            torch.from_numpy(stacked_state).float().unsqueeze(0).to(self.config.device)
        )
        with torch.no_grad():
            q_values = self._model(state_tensor)[0].cpu().numpy()

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

            metrics = self._feature_builder.extract_metrics(
                game_state, unit_id, game_state.enemy_units
            )
            if unit_id in self._last_state:
                reward = self._feature_builder.compute_reward(
                    self._last_metrics.get(unit_id, {}),
                    metrics,
                    True,
                    num_enemy_units_alive,
                )
                done = 1.0 if not num_enemy_units_alive else 0.0
                self._replay_buffer.add(
                    self._last_state[unit_id],
                    head_index,
                    self._last_action[unit_id],
                    reward,
                    stacked_state,
                    done,
                )

            action_index = self._select_action(q_values[head_index])
            action_type = ActionType.from_index(action_index)
            await self._execute_action(unit_id, action_type, game_state)

            self._last_state[unit_id] = stacked_state
            self._last_action[unit_id] = action_type
            self._last_metrics[unit_id] = metrics

        self._train_step()

        if self._step_count % self.config.target_update_interval == 0:
            logger.info(f"Updating target network at step {self._step_count}")
            self._target_model.load_state_dict(self._model.state_dict())
        if self._step_count % self.config.save_interval == 0:
            logger.info(
                f"Saving model checkpoint at step {self._step_count} to {self.config.checkpoint_path}"
            )
            self._model.save(self.config.checkpoint_path)

        # update epsilon
        if self.config.epsilon_start > self.config.epsilon_min:
            new_epsilon = max(
                self.config.epsilon_min,
                self.config.epsilon_start * self.config.epsilon_decay,
            )
            logger.debug(
                f"Updating epsilon from {self.config.epsilon_start} to {new_epsilon}"
            )
            self.config.epsilon_start = new_epsilon

        await self.admin_client.send_request_tick()

    def _train_step(self) -> None:
        if len(self._replay_buffer) < self.config.batch_size:
            return

        sample = self._replay_buffer.sample(self.config.batch_size)

        states = torch.from_numpy(sample.states).float().to(self.config.device)
        next_states = (
            torch.from_numpy(sample.next_states).float().to(self.config.device)
        )
        head_indices = (
            torch.from_numpy(sample.head_indices).long().to(self.config.device)
        )
        actions = torch.from_numpy(sample.actions).long().to(self.config.device)
        rewards = torch.from_numpy(sample.rewards).float().to(self.config.device)
        dones = torch.from_numpy(sample.dones).float().to(self.config.device)

        q_values = self._model(states)
        q_selected = q_values[
            torch.arange(self.config.batch_size), head_indices, actions
        ]

        with torch.no_grad():
            q_next = self._target_model(next_states)
            max_q_next = torch.max(q_next, dim=2).values
            max_q = max_q_next[torch.arange(self.config.batch_size), head_indices]
            targets = rewards + (1.0 - dones) * self.config.gamma * max_q

        loss = torch.mean((q_selected - targets) ** 2)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()

        if self._step_count % 100 == 0:
            logger.info(
                "Training step %s, loss %.4f, epsilon %.3f",
                self._step_count,
                loss.item(),
                self.config.epsilon_start,
            )

    def _select_action(self, q_values: np.ndarray) -> int:
        if (p := random.random()) < self.config.epsilon_start:
            logger.debug(
                f"Selecting random action due to {p=} < epsilon={self.config.epsilon_start:.3f}",
            )
            return random.choice(list(range(NUM_ACTIONS)))
        logger.debug("Selecting greedy action")
        return max(list(range(NUM_ACTIONS)), key=lambda idx: q_values[idx])

    async def _execute_action(
        self, unit_id: str, action_type: ActionType, game_state: TypedGameState
    ) -> None:
        logger.debug(f"Executing action {action_type} for unit {unit_id}")
        action_packet = action_type.to_action_packet(unit_id, game_state)
        if isinstance(action_packet, SkipAction):
            logger.debug(f"Skipping action for unit {unit_id}, because {action_type=}")
            return
        else:
            logger.debug(
                f"Sending action packet {action_packet} from {action_type=} for unit {unit_id}"
            )
        await self.agent_client._send(action_packet.to_dict())

    async def _on_endgame(self, *args, **kwargs) -> None:
        logger.info("Endgame received, resetting trainer state")
        logger.info(f"{args=}, {kwargs=}")
        self._feature_builder.reset_stack()
        self._last_state.clear()
        self._last_action.clear()
        self._last_metrics.clear()
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
