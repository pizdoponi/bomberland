import asyncio
import logging
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from dqn_config import DQNConfig
from dqn_model import NUM_ACTIONS, ActionType, DQNModel, ReplayBuffer
from dqn_shared import DQNFeatureBuilder, action_to_move
from game_state import GameState
from torch.optim.adamw import AdamW
from types_ import GameState as TypedGameState

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

        self.config = DQNConfig.from_env()

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
        agent_connection = await self.agent_client.connect()
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
        await asyncio.gather(admin_task, agent_task, kickoff_task)

    async def _on_game_tick(self, tick_number: int, game_state: Dict):
        if not self._first_tick_event.is_set():
            self._first_tick_event.set()

        self._step_count += 1

        typed_state = TypedGameState.from_dict(game_state)
        my_units = typed_state.my_units
        my_units_sorted = sorted([unit.unit_id for unit in my_units])

        enemy_units = typed_state.enemy_units

        if self._model is None:
            in_channels = (
                self._feature_builder.num_channels * self.config.frame_stack_size
            )
            self._model = DQNModel(
                conv_in_channels=in_channels,
                conv_hidden_channels=self.config.conv_hidden_channels,
                conv_out_channels=self.config.conv_out_channels,
                height=typed_state.world.height,
                width=typed_state.world.width,
                num_heads=self._feature_builder.num_heads,
                num_actions=NUM_ACTIONS,
                fc_hidden_dim=self.config.hidden_dim,
            ).to(self.config.device)
            self._target_model = DQNModel(
                conv_in_channels=in_channels,
                conv_hidden_channels=self.config.conv_hidden_channels,
                conv_out_channels=self.config.conv_out_channels,
                height=typed_state.world.height,
                width=typed_state.world.width,
                num_heads=self._feature_builder.num_heads,
                num_actions=NUM_ACTIONS,
                fc_hidden_dim=self.config.hidden_dim,
            ).to(self.config.device)
            self._optimizer = AdamW(
                self._model.parameters(), lr=self.config.learning_rate
            )
            if os.path.exists(self.config.load_path):
                self._model.load(self.config.load_path)
                self._target_model.load_state_dict(self._model.state_dict())
                logger.info("Loaded checkpoint from %s", self.config.load_path)

        frame = self._feature_builder.encode_frame(typed_state)
        stacked_state = self._feature_builder.update_frame_stack(frame)

        cache = self._feature_builder.build_cache(typed_state)
        enemy_alive_units = [unit.unit_id for unit in enemy_units if unit.is_alive()]
        enemy_alive = len(enemy_alive_units) > 0

        for unit_id in list(self._last_state.keys()):
            unit_state = typed_state.get_unit(unit_id)
            unit_alive = unit_state.is_alive() if unit_state else False
            if not unit_alive:
                current_metrics = self._feature_builder.extract_metrics(
                    typed_state, unit_id, enemy_units
                )
                reward = self._feature_builder.compute_reward(
                    self._last_metrics.get(unit_id, {}),
                    current_metrics,
                    False,
                    enemy_alive,
                )
                head_index = self._feature_builder.unit_to_head_index(
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
            unit_state = typed_state.get_unit(unit_id)
            if unit_state is None or not unit_state.is_alive():
                continue

            head_index = self._feature_builder.unit_to_head_index(
                unit_id, my_units_sorted
            )
            if head_index is None:
                continue

            metrics = self._feature_builder.extract_metrics(
                typed_state, unit_id, enemy_units
            )
            if unit_id in self._last_state:
                reward = self._feature_builder.compute_reward(
                    self._last_metrics.get(unit_id, {}),
                    metrics,
                    True,
                    enemy_alive,
                )
                done = 1.0 if not enemy_alive else 0.0
                self._replay_buffer.add(
                    self._last_state[unit_id],
                    head_index,
                    self._last_action[unit_id],
                    reward,
                    stacked_state,
                    done,
                )

            legal_actions = self._feature_builder.legal_actions(
                typed_state, unit_id, cache
            )
            action_index = self._select_action(q_values[head_index], legal_actions)
            action_type = ActionType.from_index(action_index)
            await self._execute_action(unit_id, action_type, cache.team_bombs)

            self._last_state[unit_id] = stacked_state
            self._last_action[unit_id] = action_type
            self._last_metrics[unit_id] = metrics

        self._train_step()
        if self._step_count % self.config.target_update_interval == 0:
            self._target_model.load_state_dict(self._model.state_dict())
        if self._step_count % self.config.save_interval == 0:
            self._model.save(self.config.checkpoint_path)

        if self.config.epsilon_start > self.config.epsilon_min:
            self.config.epsilon_start = max(
                self.config.epsilon_min,
                self.config.epsilon_start * self.config.epsilon_decay,
            )

        await self.admin_client.send_request_tick()

    def _train_step(self) -> None:
        if len(self._replay_buffer) < self.config.batch_size:
            return
        states, head_indices, actions, rewards, next_states, dones = (
            self._replay_buffer.sample(self.config.batch_size)
        )
        states_t = torch.from_numpy(states).float().to(self.config.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.config.device)
        head_idx_t = torch.from_numpy(head_indices).long().to(self.config.device)
        actions_t = torch.from_numpy(actions).long().to(self.config.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.config.device)
        dones_t = torch.from_numpy(dones).float().to(self.config.device)

        q_values = self._model(states_t)
        q_selected = q_values[
            torch.arange(self.config.batch_size), head_idx_t, actions_t
        ]

        with torch.no_grad():
            q_next = self._target_model(next_states_t)
            max_q_next = torch.max(q_next, dim=2).values
            max_q = max_q_next[torch.arange(self.config.batch_size), head_idx_t]
            targets = rewards_t + (1.0 - dones_t) * self.config.gamma * max_q

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

    def _select_action(self, q_values: np.ndarray, legal_actions: List[int]) -> int:
        if not legal_actions:
            logger.warning("No legal actions available, defaulting to NOOP")
            return ActionType.NOOP.value
        if random.random() < self.config.epsilon_start:
            logger.debug(
                "Selecting random action due to epsilon %.3f", self.config.epsilon_start
            )
            return random.choice(legal_actions)
        return max(legal_actions, key=lambda idx: q_values[idx])

    async def _execute_action(
        self, unit_id: str, action_type: ActionType, team_bombs: List[Tuple[int, int]]
    ) -> None:
        move = action_to_move(action_type)

        if move is not None:
            await self.agent_client.send_move(move, unit_id)
        elif action_type == ActionType.PLACE_BOMB:
            await self.agent_client.send_bomb(unit_id)
        elif action_type == ActionType.DETONATE_BOMB:
            if team_bombs:
                x, y = team_bombs[0]
                await self.agent_client.send_detonate(x, y, unit_id)
        elif action_type == ActionType.NOOP:
            return
        else:
            logger.warning("Unhandled action %s for unit %s", action_type, unit_id)

    async def _on_endgame(self, payload: Dict) -> None:
        if self.admin_client is None:
            return
        self._feature_builder.reset_stack()
        self._last_state.clear()
        self._last_action.clear()
        self._last_metrics.clear()
        world_seed = random.randint(0, 2**31 - 1)
        prng_seed = random.randint(0, 2**31 - 1)
        await self.admin_client.send_request_game_reset(
            world_seed=world_seed, prng_seed=prng_seed
        )
        await self.admin_client.send_request_tick()

    async def _ensure_first_tick(self) -> None:
        # Initial request_tick can be ignored if the game hasn't started yet.
        while not self._first_tick_event.is_set():
            await self.admin_client.send_request_tick()
            await asyncio.sleep(0.5)


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
