from __future__ import annotations

from collections import deque
from typing import Deque, List

import numpy as np
from dqn_config import DQNConfig
from dqn_model import ActionType
from types_ import EntityType, UnitState
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

DETONATION_INDEX_BY_ACTION = {
    ActionType.DETONATE_BOMB_0: 0,
    ActionType.DETONATE_BOMB_1: 1,
    ActionType.DETONATE_BOMB_2: 2,
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

    def unit_id_to_head_index(self, unit_id: str, units: List[str]) -> int:
        """Get the head index for a given unit_id based on its position in the units list.

        Args:
            unit_id: The unique identifier of the unit.
            units: List of unit IDs. The list is sorted to ensure consistent indexing.

        Returns:
            The index of the unit_id in the sorted units list, which corresponds to the head index.

        Raises:
            ValueError: If the unit_id is not found in the units list or if its index exceeds num_heads.
        """
        if unit_id not in units:
            raise ValueError(f"unit_id {unit_id} not in units list {units}")
        sorted_units = sorted(units)
        idx = sorted_units.index(unit_id)
        if idx >= self.num_heads:
            raise ValueError(
                f"unit_id {unit_id} index {idx} exceeds num_heads {self.num_heads}"
            )
        return idx

    def compute_team_and_unit_rewards(
        self,
        prev_game_state: TypedGameState,
        curr_game_state: TypedGameState,
        actions_taken: dict[str, ActionType],
    ) -> tuple[float, dict[str, float], bool]:
        """
        Returns:
            team_reward: shared objective reward
            unit_rewards: per-unit reward = team_reward + small PBRS-like shaping
            done: terminal flag (enemy dead or me dead)
        """

        # ---------- Terminal ----------
        prev_enemy_alive = len(prev_game_state.enemy_alive_units)
        prev_my_alive = len(prev_game_state.my_alive_units)
        curr_enemy_alive = len(curr_game_state.enemy_alive_units)
        curr_my_alive = len(curr_game_state.my_alive_units)

        done = (curr_enemy_alive == 0) or (curr_my_alive == 0)

        terminal = 0.0
        if curr_enemy_alive == 0 and curr_my_alive > 0:
            terminal = 1.0
        elif curr_my_alive == 0 and curr_enemy_alive > 0:
            terminal = -1.0
        elif curr_my_alive == 0 and curr_enemy_alive == 0:
            terminal = 0.0

        # ---------- Dense objective: HP swing + death swing ----------
        prev_my_hp = sum(u.hp for u in prev_game_state.my_units)
        curr_my_hp = sum(u.hp for u in curr_game_state.my_units)
        prev_enemy_hp = sum(u.hp for u in prev_game_state.enemy_units)
        curr_enemy_hp = sum(u.hp for u in curr_game_state.enemy_units)

        enemy_hp_lost = max(0, prev_enemy_hp - curr_enemy_hp)
        my_hp_lost = max(0, prev_my_hp - curr_my_hp)

        enemy_deaths = max(0, prev_enemy_alive - curr_enemy_alive)
        my_deaths = max(0, prev_my_alive - curr_my_alive)

        # Scale to keep values stable
        enemy_hp_reward = 0.10 * (enemy_hp_lost / 9.0)  # 9 = 3 units * 3 HP
        my_hp_penalty = 0.20 * (my_hp_lost / 9.0)
        hp_term = enemy_hp_reward - my_hp_penalty
        death_term = 0.30 * ((enemy_deaths - my_deaths) / 3.0)  # 3 units per agent

        step_penalty = 0.0 if done else -0.001

        team_reward = terminal + hp_term + death_term + step_penalty

        # ---------- Per-unit shaping (small): safety + enemy proximity ----------
        def danger_score(unit: UnitState) -> float:
            if curr_game_state.is_dangerous_tile(unit.x, unit.y):
                return 1.0
            return 0.0

        def enemy_proximity(unit: UnitState) -> float:
            enemies = curr_game_state.enemy_alive_units
            if not enemies:
                return 1.0
            dmin = min(unit.position.distance_to(e.position) for e in enemies)
            return 1.0 / (1.0 + float(dmin))

        def proactivity(unit: UnitState) -> float:
            action = actions_taken.get(unit.unit_id)
            if action is None:
                return 0.0
            if action == ActionType.NOOP:
                return -0.001
            if action == ActionType.PLACE_BOMB:
                return 0.002
            if action.is_bomb_detonation():
                bomb_index = DETONATION_INDEX_BY_ACTION.get(action)
                if bomb_index is None:
                    return 0.0
                detonated_bombs = prev_game_state.my_units_bombs(
                    unit_id=unit.unit_id, bomb_idx=bomb_index
                )
                if not detonated_bombs:
                    return 0.0
                blast_tiles = prev_game_state.get_blast_tiles_if_detonated(
                    detonated_bombs[0].position,
                    require_armed=False,
                )
                blocks_hit = sum(
                    1
                    for point in blast_tiles
                    for entity in prev_game_state.entities_at(point.x, point.y)
                    if entity.entity_type
                    in {
                        EntityType.WOOD_BLOCK,
                        EntityType.ORE_BLOCK,
                    }
                )
                units_hit = sum(
                    1
                    for enemy in prev_game_state.enemy_alive_units
                    if enemy.position in blast_tiles
                )
                return 0.1 * blocks_hit + 0.3 * units_hit
            return 0.0

        def stupidity(unit: UnitState) -> float:
            action = actions_taken.get(unit.unit_id)
            if action is None:
                return 0.0
            previous_unit_state = prev_game_state.get_unit(unit.unit_id) or unit
            if action.is_movement():
                dx, dy = MOVE_DELTAS[action]
                new_x = previous_unit_state.x + dx
                new_y = previous_unit_state.y + dy
                if curr_game_state.is_dangerous_tile(new_x, new_y):
                    return 0.3
            if action.is_bomb_detonation() and (curr_my_hp < prev_my_hp):
                # you don't detonate bombs if it hurts you stoopid
                return 1.0
            return 0.0

        # Potential: higher is better
        def potential(unit: UnitState) -> float:
            if not unit.is_alive():
                return 0.0
            safe = 1.0 - 0.7 * stupidity(unit) - 0.3 * danger_score(unit)
            press = 0.2 * enemy_proximity(unit) + 0.8 * proactivity(unit)
            return 0.65 * safe + 0.35 * press

        unit_rewards: dict[str, float] = {}
        gamma = self.config.gamma
        shaping_weight = 0.1

        for unit in curr_game_state.my_alive_units:
            # PBRS-style: gamma*Phi(s') - Phi(s)
            phi_prev = potential(prev_game_state.get_unit(unit.unit_id) or unit)
            phi_curr = potential(unit)
            shaping = gamma * phi_curr - phi_prev
            unit_rewards[unit.unit_id] = team_reward + shaping_weight * shaping

        return team_reward, unit_rewards, done
