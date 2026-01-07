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
        Compute rewards for team and individual units.

        Returns:
            team_reward: shared objective reward
            unit_rewards: per-unit reward = team_reward + small PBRS-like shaping
            done: terminal flag (enemy dead or me dead)
        """

        # ---------- Terminal conditions ----------
        prev_enemy_alive = len(prev_game_state.enemy_alive_units)
        prev_my_alive = len(prev_game_state.my_alive_units)
        curr_enemy_alive = len(curr_game_state.enemy_alive_units)
        curr_my_alive = len(curr_game_state.my_alive_units)

        done = (curr_enemy_alive == 0) or (curr_my_alive == 0)

        # Terminal reward: big signal for winning/losing
        terminal = 0.0
        if curr_enemy_alive == 0 and curr_my_alive > 0:
            terminal = 5.0  # Win
        elif curr_my_alive == 0 and curr_enemy_alive > 0:
            terminal = -5.0  # Loss
        elif curr_my_alive == 0 and curr_enemy_alive == 0:
            terminal = -1.0  # Both dead - slight penalty (we want to survive)

        # ---------- Dense rewards: HP changes and kills ----------
        prev_my_hp = sum(u.hp for u in prev_game_state.my_units)
        curr_my_hp = sum(u.hp for u in curr_game_state.my_units)
        prev_enemy_hp = sum(u.hp for u in prev_game_state.enemy_units)
        curr_enemy_hp = sum(u.hp for u in curr_game_state.enemy_units)

        enemy_hp_lost = max(0, prev_enemy_hp - curr_enemy_hp)
        my_hp_lost = max(0, prev_my_hp - curr_my_hp)

        enemy_deaths = max(0, prev_enemy_alive - curr_enemy_alive)
        my_deaths = max(0, prev_my_alive - curr_my_alive)

        # Reward for damaging enemies (scaled to be meaningful but not overwhelming)
        enemy_damage_reward = 0.2 * enemy_hp_lost  # +0.2 per HP damage to enemy
        my_damage_penalty = 0.3 * my_hp_lost  # -0.3 per HP lost (asymmetric to encourage survival)

        # Reward for kills
        kill_reward = 0.5 * enemy_deaths  # +0.5 per enemy killed
        death_penalty = 0.7 * my_deaths  # -0.7 per ally death

        # Small step penalty to encourage efficiency (but not too large to overshadow other signals)
        step_penalty = -0.002 if not done else 0.0

        team_reward = (
            terminal
            + enemy_damage_reward
            - my_damage_penalty
            + kill_reward
            - death_penalty
            + step_penalty
        )

        # ---------- Per-unit shaping rewards ----------
        def danger_score(unit: UnitState) -> float:
            """Returns 1.0 if unit is on a dangerous tile, 0.0 otherwise."""
            if curr_game_state.is_dangerous_tile(unit.x, unit.y):
                return 1.0
            return 0.0

        def enemy_proximity(unit: UnitState) -> float:
            """Returns higher value when closer to enemies (encourages aggression)."""
            enemies = curr_game_state.enemy_alive_units
            if not enemies:
                return 1.0
            dmin = min(unit.position.distance_to(e.position) for e in enemies)
            return 1.0 / (1.0 + float(dmin))

        def action_quality(unit: UnitState) -> float:
            """Evaluate action quality for shaping."""
            action = actions_taken.get(unit.unit_id)
            if action is None:
                return 0.0

            # Slight penalty for doing nothing when alive
            if action == ActionType.NOOP:
                return -0.01

            # Small reward for placing bombs (encourages activity)
            if action == ActionType.PLACE_BOMB:
                return 0.02

            # Reward for useful detonations
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
                # Count what we hit
                blocks_hit = sum(
                    1
                    for point in blast_tiles
                    for entity in prev_game_state.entities_at(point.x, point.y)
                    if entity.entity_type
                    in {EntityType.WOOD_BLOCK, EntityType.ORE_BLOCK}
                )
                units_hit = sum(
                    1
                    for enemy in prev_game_state.enemy_alive_units
                    if enemy.position in blast_tiles
                )
                return 0.05 * blocks_hit + 0.15 * units_hit

            return 0.0

        def movement_penalty(unit: UnitState) -> float:
            """Penalty for moving into danger or self-destructive detonations."""
            action = actions_taken.get(unit.unit_id)
            if action is None:
                return 0.0

            previous_unit_state = prev_game_state.get_unit(unit.unit_id) or unit

            # Penalty for moving into dangerous tiles
            if action.is_movement():
                dx, dy = MOVE_DELTAS[action]
                new_x = previous_unit_state.x + dx
                new_y = previous_unit_state.y + dy
                if curr_game_state.is_dangerous_tile(new_x, new_y):
                    return 0.5  # Significant penalty for walking into danger

            # Large penalty for detonating bombs that hurt yourself
            if action.is_bomb_detonation() and (curr_my_hp < prev_my_hp):
                return 1.0

            return 0.0

        def unit_potential(unit: UnitState) -> float:
            """
            Potential function for PBRS-style shaping.
            Higher values indicate better states.
            """
            if not unit.is_alive():
                return 0.0

            # Safety component (avoid danger)
            safety = 1.0 - danger_score(unit) - movement_penalty(unit)

            # Aggression component (be near enemies, take useful actions)
            aggression = 0.3 * enemy_proximity(unit) + 0.7 * action_quality(unit)

            # Balance safety and aggression
            return max(0.0, 0.6 * safety + 0.4 * aggression)

        # Compute per-unit rewards
        unit_rewards: dict[str, float] = {}
        gamma = self.config.gamma
        shaping_weight = 0.15  # Weight for shaping rewards

        for unit in curr_game_state.my_alive_units:
            # PBRS: gamma * Phi(s') - Phi(s)
            prev_unit = prev_game_state.get_unit(unit.unit_id) or unit
            phi_prev = unit_potential(prev_unit)
            phi_curr = unit_potential(unit)
            shaping = gamma * phi_curr - phi_prev
            unit_rewards[unit.unit_id] = team_reward + shaping_weight * shaping

        return team_reward, unit_rewards, done
