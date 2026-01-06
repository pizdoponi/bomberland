from __future__ import annotations

from collections import deque
from typing import Deque, List

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
        hp_term = 0.10 * ((enemy_hp_lost - my_hp_lost) / 9.0)  # 9 = 3 units * 3 HP
        death_term = 0.30 * ((enemy_deaths - my_deaths) / 3.0)  # 3 units per agent

        step_penalty = 0.0 if done else -0.001

        team_reward = terminal + hp_term + death_term + step_penalty

        # ---------- Per-unit shaping (small): safety + enemy proximity ----------
        # This is intentionally simple (no full blast simulation).
        def danger_score(gs: TypedGameState, unit) -> float:
            # Standing on blast is worst
            if gs.is_dangerous_tile(unit.x, unit.y):
                # bomb OR blast present
                return 1.0

            # If you're near a blast tile, also bad
            blasts = gs.entities_of_type(EntityType.BLAST)
            if blasts:
                dmin = min(abs(unit.x - b.x) + abs(unit.y - b.y) for b in blasts)
                if dmin == 1:
                    return 0.7
                if dmin == 2:
                    return 0.3
            return 0.0

        def enemy_proximity(gs: TypedGameState, unit) -> float:
            enemies = [e for e in gs.enemy_units if e.is_alive()]
            if not enemies:
                return 1.0
            dmin = min(unit.position.distance_to(e.position) for e in enemies)
            return 1.0 / (1.0 + float(dmin))

        # Potential: higher is better
        def potential(gs: TypedGameState, unit) -> float:
            if not unit.is_alive():
                return 0.0
            safe = 1.0 - danger_score(gs, unit)
            press = enemy_proximity(gs, unit)
            return 0.65 * safe + 0.35 * press

        unit_rewards: dict[str, float] = {}
        gamma = self.config.gamma
        shaping_weight = 0.05

        for u in curr_game_state.my_units:
            if not u.is_alive():
                continue
            # PBRS-style: gamma*Phi(s') - Phi(s)
            phi_prev = potential(prev_game_state, prev_game_state.get_unit(u.unit_id) or u)
            phi_curr = potential(curr_game_state, u)
            shaping = gamma * phi_curr - phi_prev
            unit_rewards[u.unit_id] = team_reward + shaping_weight * shaping

        return team_reward, unit_rewards, done

