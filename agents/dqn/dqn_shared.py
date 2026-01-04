from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

import numpy as np
from dqn_config import DQNConfig
from dqn_model import ActionType
from types_ import EntityType
from types_ import GameState as TypedGameState

MOVE_DELTAS = {
    ActionType.UP: (0, 1),
    ActionType.RIGHT: (1, 0),
    ActionType.DOWN: (0, -1),
    ActionType.LEFT: (-1, 0),
}

MOVE_BY_ACTION = {
    ActionType.UP: "up",
    ActionType.RIGHT: "right",
    ActionType.DOWN: "down",
    ActionType.LEFT: "left",
}


class DQNFeatureBuilder:
    def __init__(self, config: DQNConfig):
        self.config = config
        self.num_channels = 17
        self.num_heads = 3
        self._frame_stack: Deque[np.ndarray] = deque(
            maxlen=self.config.frame_stack_size
        )

    def reset_stack(self) -> None:
        self._frame_stack.clear()

    def update_frame_stack(self, frame: np.ndarray) -> np.ndarray:
        if len(self._frame_stack) == 0:
            for _ in range(self.config.frame_stack_size):
                self._frame_stack.append(frame)
        else:
            self._frame_stack.append(frame)
        stacked = np.concatenate(list(self._frame_stack), axis=0)
        return stacked

    def encode_frame(self, game_state: TypedGameState) -> np.ndarray:
        height = game_state.world.height
        width = game_state.world.width
        frame = np.zeros((self.num_channels, height, width), dtype=np.float32)

        for entity in game_state.entities:
            x, y = entity.x, entity.y
            if entity.entity_type == EntityType.METAL_BLOCK:
                frame[0, y, x] = 1.0
            elif entity.entity_type == EntityType.ORE_BLOCK:
                hp = float(entity.hp or self.config.max_ore_hp)
                frame[1, y, x] = min(hp / self.config.max_ore_hp, 1.0)
            elif entity.entity_type == EntityType.WOOD_BLOCK:
                frame[2, y, x] = 1.0
            elif entity.entity_type == EntityType.BLAST:
                frame[3, y, x] = entity.time_until_expires(game_state.tick) or 0.0
            elif entity.entity_type in {
                EntityType.BLAST_POWERUP,
                EntityType.FREEZE_POWERUP,
            }:
                frame[4, y, x] = entity.time_until_expires(game_state.tick) or 0.0
            elif entity.entity_type == EntityType.BOMB:
                bomb_owner_id_to_frame_index = {
                    "c": 5,
                    "d": 6,
                    "e": 7,
                    "f": 8,
                    "g": 9,
                    "h": 10,
                }
                bomb_owner = entity.owner_unit_id
                assert bomb_owner is not None, "Bomb owner_unit_id should not be None"
                frame_index = bomb_owner_id_to_frame_index[bomb_owner]
                frame[frame_index, y, x] = (
                    entity.time_until_expires(game_state.tick) or 0.0
                )

        for i, my_unit in enumerate(game_state.my_units):
            frame[11 + i, my_unit.y, my_unit.x] = my_unit.hp / self.config.max_unit_hp

        for i, enemy_unit in enumerate(game_state.enemy_units):
            frame[14 + i, enemy_unit.y, enemy_unit.x] = (
                enemy_unit.hp / self.config.max_unit_hp
            )

        return frame

    def unit_id_to_head_index(
        self, unit_id: str, sorted_units: List[str]
    ) -> int | None:
        if unit_id not in sorted_units:
            return None
        idx = sorted_units.index(unit_id)
        if idx >= self.num_heads:
            return None
        return idx

    def extract_metrics(
        self, game_state: TypedGameState, unit_id: str, enemy_units: List
    ) -> Dict[str, float]:
        unit = game_state.get_unit(unit_id)
        hp = float(unit.hp) if unit else 0.0
        enemy_hp = float(sum(enemy.hp for enemy in enemy_units))
        return {"hp": hp, "enemy_hp": enemy_hp}

    def compute_reward(
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

    def _is_walkable(
        self, x: int, y: int, width: int, height: int, blocked_positions: set
    ) -> bool:
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        if (x, y) in blocked_positions:
            return False
        return True
