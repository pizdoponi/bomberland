import asyncio
import logging
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from dqn_config import DQNConfig
from dqn_model import DQNModel, ReplayBuffer
from dqn_shared import ACTIONS, DQNFeatureBuilder
from game_state import GameState
from types_ import GameState as TypedGameState

logging.basicConfig(
    level=logging.INFO,
    format="[dqn-train] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

uri = os.environ.get(
    "GAME_CONNECTION_STRING"
) or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"


class DQNTrainer:
    def __init__(self) -> None:
        self._client = GameState(uri)
        self._client.set_game_tick_callback(self._on_game_tick)

        self.config = DQNConfig.from_env()
        self._epsilon = self.config.epsilon_start

        self._device = torch.device(self.config.device)

        self._feature_builder = DQNFeatureBuilder(self.config)

        self._model = None
        self._target_model = None
        self._optimizer = None
        self._replay_buffer = ReplayBuffer(self.config.replay_capacity)

        self._last_state: Dict[str, np.ndarray] = {}
        self._last_action: Dict[str, int] = {}
        self._last_metrics: Dict[str, Dict[str, float]] = {}
        self._step_count = 0

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [asyncio.ensure_future(self._client._handle_messages(connection))]
        loop.run_until_complete(asyncio.wait(tasks))

    async def _on_game_tick(self, tick_number: int, game_state: Dict):
        self._step_count += 1

        typed_state = TypedGameState.from_dict(game_state)
        my_units = typed_state.my_units
        my_units_sorted = sorted([unit.unit_id for unit in my_units])

        enemy_units = typed_state.enemy_units

        if self._model is None:
            in_channels = self._feature_builder.channels * self.config.frame_stack_size
            self._model = DQNModel(
                in_channels=in_channels,
                height=typed_state.world.height,
                width=typed_state.world.width,
                num_heads=self._feature_builder.num_heads,
                num_actions=len(ACTIONS),
                hidden_dim=self.config.hidden_dim,
            ).to(self._device)
            self._target_model = DQNModel(
                in_channels=in_channels,
                height=typed_state.world.height,
                width=typed_state.world.width,
                num_heads=self._feature_builder.num_heads,
                num_actions=len(ACTIONS),
                hidden_dim=self.config.hidden_dim,
            ).to(self._device)
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(), lr=self.config.learning_rate
            )
            if os.path.exists(self.config.load_path):
                self._model.load(self.config.load_path)
                self._target_model.load_state_dict(self._model.state_dict())
                logger.info("Loaded checkpoint from %s", self.config.load_path)

        frame = self._feature_builder.encode_frame(
            typed_state, typed_state.my_agent_id
        )
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
            torch.from_numpy(stacked_state).float().unsqueeze(0).to(self._device)
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
            await self._execute_action(unit_id, action_index, cache.team_bombs)

            self._last_state[unit_id] = stacked_state
            self._last_action[unit_id] = action_index
            self._last_metrics[unit_id] = metrics

        self._train_step()
        if self._step_count % self.config.target_update_interval == 0:
            self._target_model.load_state_dict(self._model.state_dict())
        if self._step_count % self.config.save_interval == 0:
            self._model.save(self.config.checkpoint_path)

        if self._epsilon > self.config.epsilon_min:
            self._epsilon = max(
                self.config.epsilon_min, self._epsilon * self.config.epsilon_decay
            )

    def _train_step(self) -> None:
        if len(self._replay_buffer) < self.config.batch_size:
            return
        states, head_indices, actions, rewards, next_states, dones = (
            self._replay_buffer.sample(self.config.batch_size)
        )
        states_t = torch.from_numpy(states).float().to(self._device)
        next_states_t = torch.from_numpy(next_states).float().to(self._device)
        head_idx_t = torch.from_numpy(head_indices).long().to(self._device)
        actions_t = torch.from_numpy(actions).long().to(self._device)
        rewards_t = torch.from_numpy(rewards).float().to(self._device)
        dones_t = torch.from_numpy(dones).float().to(self._device)

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
                self._epsilon,
            )

    def _select_action(self, q_values: np.ndarray, legal_actions: List[int]) -> int:
        if not legal_actions:
            return ACTIONS.index("wait")
        if random.random() < self._epsilon:
            return random.choice(legal_actions)
        return max(legal_actions, key=lambda idx: q_values[idx])

    async def _execute_action(
        self, unit_id: str, action_index: int, team_bombs: List[Tuple[int, int]]
    ) -> None:
        action = ACTIONS[action_index]

        if action in {"up", "down", "left", "right"}:
            await self._client.send_move(action, unit_id)
        elif action == "bomb":
            await self._client.send_bomb(unit_id)
        elif action == "detonate":
            if team_bombs:
                x, y = team_bombs[0]
                await self._client.send_detonate(x, y, unit_id)
        elif action == "wait":
            return
        else:
            logger.warning("Unhandled action %s for unit %s", action, unit_id)


def main() -> None:
    for _ in range(0, 10):
        while True:
            try:
                DQNTrainer()
            except Exception as exc:
                logger.error("Trainer error: %s", exc)
                time.sleep(5)
                continue
            break


if __name__ == "__main__":
    main()
