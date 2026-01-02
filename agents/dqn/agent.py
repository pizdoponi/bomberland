import asyncio
import logging
import os
import random
import time
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
from dqn_config import DQNConfig
from dqn_model import DQNModel, ReplayBuffer
from game_state import GameState
from types_ import AgentId, EntityType
from types_ import GameState as TypedGameState

logging.basicConfig(
    level=logging.INFO,
    format="[dqn] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

uri = os.environ.get(
    "GAME_CONNECTION_STRING"
) or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"

MOVE_ACTIONS = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


class DQNAgent:
    def __init__(self):
        self._client = GameState(uri)
        self._client.set_game_tick_callback(self._on_game_tick)

        self._actions = ["up", "down", "left", "right", "bomb", "detonate", "wait"]

        self.config = DQNConfig.from_env()
        self.config.epsilon_start = self.config.epsilon_start

        self._channels = 8
        self._num_heads = 3
        self._device = torch.device(self.config.device)

        self._model = None
        self._target_model = None
        self._optimizer = None
        self._replay_buffer = ReplayBuffer(self.config.replay_capacity)

        self._frame_stack: Deque[np.ndarray] = deque(
            maxlen=self.config.frame_stack_size
        )
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
        my_agent_id = typed_state.my_agent_id
        my_units = typed_state.my_units
        my_units_sorted = sorted([unit.unit_id for unit in my_units])

        enemy_units = typed_state.enemy_units

        if self._model is None:
            in_channels = self._channels * self.config.frame_stack_size
            self._model = DQNModel(
                in_channels=in_channels,
                height=typed_state.world.height,
                width=typed_state.world.width,
                num_heads=self._num_heads,
                num_actions=len(self._actions),
                hidden_dim=self.config.hidden_dim,
            ).to(self._device)
            self._target_model = DQNModel(
                in_channels=in_channels,
                height=typed_state.world.height,
                width=typed_state.world.width,
                num_heads=self._num_heads,
                num_actions=len(self._actions),
                hidden_dim=self.config.hidden_dim,
            ).to(self._device)
            self._optimizer = torch.optim.adamw.AdamW(
                self._model.parameters(), lr=self.config.learning_rate
            )
            if os.path.exists(self.config.load_path):
                self._model.load(self.config.load_path)
                self._target_model.load_state_dict(self._model.state_dict())
                logger.info("Loaded checkpoint from %s", self.config.load_path)

        frame = self._encode_frame(typed_state, my_agent_id)
        stacked_state = self._update_frame_stack(frame)

        blocked_positions, team_bombs = self._build_cache(typed_state)
        enemy_alive_units = [unit.unit_id for unit in enemy_units if unit.is_alive()]
        enemy_alive = len(enemy_alive_units) > 0

        for unit_id in list(self._last_state.keys()):
            unit_state = typed_state.get_unit(unit_id)
            unit_alive = unit_state.is_alive() if unit_state else False
            if not unit_alive:
                current_metrics = self._extract_metrics(
                    typed_state, unit_id, enemy_units
                )
                reward = self._compute_reward(
                    self._last_metrics.get(unit_id, {}),
                    current_metrics,
                    False,
                    enemy_alive,
                )
                head_index = self._unit_to_head_index(unit_id, my_units_sorted)
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

        state_tensor = torch.from_numpy(stacked_state).float().unsqueeze(0).to(self._device)
        with torch.no_grad():
            q_values = self._model(state_tensor)[0].cpu().numpy()

        for unit_id in my_units_sorted:
            unit_state = typed_state.get_unit(unit_id)
            if unit_state is None or not unit_state.is_alive():
                continue

            head_index = self._unit_to_head_index(unit_id, my_units_sorted)
            if head_index is None:
                continue

            metrics = self._extract_metrics(typed_state, unit_id, enemy_units)
            if unit_id in self._last_state:
                reward = self._compute_reward(
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

            legal_actions = self._legal_actions(
                typed_state, unit_id, blocked_positions, team_bombs
            )
            action_index = self._select_action(q_values[head_index], legal_actions)
            await self._execute_action(unit_id, action_index, team_bombs)

            self._last_state[unit_id] = stacked_state
            self._last_action[unit_id] = action_index
            self._last_metrics[unit_id] = metrics

        if self.config.training_enabled:
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
                self.config.epsilon_start,
            )

    def _select_action(self, q_values: np.ndarray, legal_actions: List[int]) -> int:
        if not legal_actions:
            return self._actions.index("wait")
        if self.config.training_enabled and random.random() < self.config.epsilon_start:
            return random.choice(legal_actions)
        return max(legal_actions, key=lambda idx: q_values[idx])

    async def _execute_action(
        self, unit_id: str, action_index: int, team_bombs: List[Tuple[int, int]]
    ) -> None:
        action = self._actions[action_index]

        if action in MOVE_ACTIONS:
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

    def _unit_to_head_index(self, unit_id: str, sorted_units: List[str]) -> int | None:
        if unit_id not in sorted_units:
            return None
        idx = sorted_units.index(unit_id)
        if idx >= self._num_heads:
            return None
        return idx

    def _update_frame_stack(self, frame: np.ndarray) -> np.ndarray:
        if len(self._frame_stack) == 0:
            for _ in range(self.config.frame_stack_size):
                self._frame_stack.append(frame)
        else:
            self._frame_stack.append(frame)
        stacked = np.concatenate(list(self._frame_stack), axis=0)
        return stacked

    def _encode_frame(self, game_state: TypedGameState, my_agent_id: AgentId) -> np.ndarray:
        width = game_state.world.width
        height = game_state.world.height
        frame = np.zeros((self._channels, height, width), dtype=np.float32)

        for entity in game_state.entities:
            x, y = entity.x, entity.y
            if entity.entity_type == EntityType.METAL_BLOCK:
                frame[0, y, x] = 1.0
            elif entity.entity_type == EntityType.ORE_BLOCK:
                hp = float(entity.hp or self.config.max_ore_hp)
                frame[1, y, x] = min(hp / self.config.max_ore_hp, 1.0)
            elif entity.entity_type == EntityType.WOOD_BLOCK:
                frame[2, y, x] = 1.0
            elif entity.entity_type == EntityType.BOMB:
                frame[3, y, x] = 1.0
            elif entity.entity_type == EntityType.BLAST:
                frame[4, y, x] = 1.0
            elif entity.entity_type in {
                EntityType.AMMO,
                EntityType.BLAST_POWERUP,
                EntityType.FREEZE_POWERUP,
            }:
                frame[7, y, x] = 1.0

        for unit in game_state.units.values():
            if not unit.is_alive():
                continue
            if unit.agent_id == my_agent_id:
                frame[5, unit.y, unit.x] = 1.0
            else:
                frame[6, unit.y, unit.x] = 1.0

        return frame

    def _build_cache(self, game_state: TypedGameState) -> Tuple[set, List[Tuple[int, int]]]:
        blocked_positions = set()
        team_bombs: List[Tuple[int, int]] = []
        my_unit_ids = {unit.unit_id for unit in game_state.my_units}

        for entity in game_state.entities:
            if entity.is_solid():
                blocked_positions.add((entity.x, entity.y))
            if entity.entity_type == EntityType.BOMB and entity.owner_unit_id in my_unit_ids:
                team_bombs.append((entity.x, entity.y))

        for unit in game_state.alive_units:
            blocked_positions.add((unit.x, unit.y))

        return blocked_positions, team_bombs

    def _legal_actions(
        self,
        game_state: TypedGameState,
        unit_id: str,
        blocked_positions: set,
        team_bombs: List[Tuple[int, int]],
    ) -> List[int]:
        unit = game_state.get_unit(unit_id)
        if unit is None:
            return [self._actions.index("wait")]
        x, y = unit.x, unit.y
        width = game_state.world.width
        height = game_state.world.height

        legal = []
        for action in ["up", "down", "left", "right"]:
            dx, dy = MOVE_ACTIONS[action]
            nx, ny = x + dx, y + dy
            if self._is_walkable(nx, ny, width, height, blocked_positions):
                legal.append(self._actions.index(action))

        bombs = float(unit.inventory.bombs)
        if bombs > 0:
            legal.append(self._actions.index("bomb"))

        if team_bombs:
            legal.append(self._actions.index("detonate"))

        legal.append(self._actions.index("wait"))
        return legal

    def _is_walkable(
        self, x: int, y: int, width: int, height: int, blocked_positions: set
    ) -> bool:
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        if (x, y) in blocked_positions:
            return False
        return True

    def _extract_metrics(
        self, game_state: TypedGameState, unit_id: str, enemy_units: List
    ) -> Dict[str, float]:
        unit = game_state.get_unit(unit_id)
        hp = float(unit.hp) if unit else 0.0
        enemy_hp = float(sum(enemy.hp for enemy in enemy_units))
        return {"hp": hp, "enemy_hp": enemy_hp}

    def _compute_reward(
        self,
        prev_metrics: Dict[str, float],
        curr_metrics: Dict[str, float],
        unit_alive: bool,
        enemy_alive: bool,
    ) -> float:
        prev_hp = prev_metrics.get("hp", curr_metrics.get("hp", 0.0))
        curr_hp = curr_metrics.get("hp", 0.0)
        prev_enemy_hp = prev_metrics.get("enemy_hp", curr_metrics.get("enemy_hp", 0.0))
        curr_enemy_hp = curr_metrics.get("enemy_hp", 0.0)

        reward = 0.0
        reward += (prev_enemy_hp - curr_enemy_hp) * 5.0
        reward += (curr_hp - prev_hp) * 2.0
        if not unit_alive and prev_hp > 0:
            reward -= 10.0
        if not enemy_alive and prev_enemy_hp > 0:
            reward += 10.0
        return reward


def main():
    for _ in range(0, 10):
        while True:
            try:
                DQNAgent()
            except Exception as exc:
                logger.error("Agent error: %s", exc)
                time.sleep(5)
                continue
            break


if __name__ == "__main__":
    main()
