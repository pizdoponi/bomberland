from __future__ import annotations

from collections import deque
from typing import Deque, List, Set

import numpy as np
from dqn_config import DQNConfig
from dqn_model import ActionType
from types_ import EntityType, Point
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
    """
    Feature encoding for the DQN agent.

    Channel layout (18 channels total):

    Terrain (3 channels):
        0: Metal blocks (indestructible walls) - binary
        1: Ore blocks (3 HP destructible) - HP normalized [0, 1]
        2: Wood blocks (1 HP destructible) - binary

    Hazards (3 channels):
        3: Active blasts - urgency (1.0 = about to expire, 0.0 = just created)
        4: Danger zone - tiles in blast range of any armed bomb - binary
        5: Enemy bombs - normalized time until expires (1.0 = about to explode)

    My bombs per unit (3 channels) - needed for detonation decisions:
        6: Unit 0's bombs (first unit alphabetically, e.g., 'c')
        7: Unit 1's bombs (e.g., 'd')
        8: Unit 2's bombs (e.g., 'e')

    Items (1 channel):
        9: Powerups (blast/freeze) - binary presence

    Units (6 channels):
        10-12: My units (3) - HP normalized, 0 if dead
        13-15: Enemy units (3) - HP normalized, 0 if dead

    Armed bomb indicators (2 channels):
        16: My armed bombs (any team bomb I could detonate if it were mine)
        17: Enemy armed bombs
    """

    def __init__(self, config: DQNConfig):
        self.config = config
        self.num_channels = 18
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
        tick = game_state.tick
        frame = np.zeros((self.num_channels, height, width), dtype=np.float32)

        # Get my unit IDs and create a mapping to channel indices
        my_unit_ids = sorted([u.unit_id for u in game_state.my_units])
        my_unit_to_channel = {uid: 6 + i for i, uid in enumerate(my_unit_ids[:3])}
        my_unit_id_set = set(my_unit_ids)

        for entity in game_state.entities:
            x, y = entity.x, entity.y

            if entity.entity_type == EntityType.METAL_BLOCK:
                # Channel 0: Metal blocks - binary
                frame[0, y, x] = 1.0

            elif entity.entity_type == EntityType.ORE_BLOCK:
                # Channel 1: Ore blocks - HP normalized
                hp = float(entity.hp or self.config.max_ore_hp)
                frame[1, y, x] = hp / self.config.max_ore_hp

            elif entity.entity_type == EntityType.WOOD_BLOCK:
                # Channel 2: Wood blocks - binary
                frame[2, y, x] = 1.0

            elif entity.entity_type == EntityType.BLAST:
                # Channel 3: Active blasts - urgency (1.0 = about to expire/safe soon)
                time_left = entity.time_until_expires(tick)
                if time_left is not None:
                    frame[3, y, x] = 1.0 - time_left  # Invert: high = expiring soon
                else:
                    frame[3, y, x] = 0.5

            elif entity.entity_type in {EntityType.BLAST_POWERUP, EntityType.FREEZE_POWERUP}:
                # Channel 9: Powerups - binary presence
                frame[9, y, x] = 1.0

            elif entity.entity_type == EntityType.BOMB:
                owner = entity.owner_unit_id
                is_my_bomb = owner in my_unit_id_set

                # Compute urgency: 1.0 = about to explode, 0.0 = just placed
                time_left = entity.time_until_expires(tick)
                urgency = (1.0 - time_left) if time_left is not None else 0.5

                if is_my_bomb:
                    # Channels 6-8: Per-unit bomb channels
                    channel = my_unit_to_channel.get(owner)
                    if channel is not None:
                        frame[channel, y, x] = urgency

                    # Channel 16: My armed bombs
                    if entity.is_armed(tick):
                        frame[16, y, x] = 1.0
                else:
                    # Channel 5: Enemy bombs
                    frame[5, y, x] = urgency

                    # Channel 17: Enemy armed bombs
                    if entity.is_armed(tick):
                        frame[17, y, x] = 1.0

        # Channel 4: Danger zone - tiles that would be hit by any armed bomb
        # Use cached bomb list instead of collecting during iteration
        danger_tiles: Set[Point] = set()
        for bomb in game_state._all_bombs or []:
            if bomb.is_armed(tick):
                blast_tiles = game_state.get_blast_tiles_if_detonated(
                    bomb.position, require_armed=False
                )
                danger_tiles.update(blast_tiles)

        for tile in danger_tiles:
            if 0 <= tile.x < width and 0 <= tile.y < height:
                frame[4, tile.y, tile.x] = 1.0

        # Channels 10-12: My units - HP normalized, only if alive
        for i, unit in enumerate(game_state.my_units):
            if i < 3 and unit.is_alive():
                frame[10 + i, unit.y, unit.x] = unit.hp / self.config.max_unit_hp

        # Channels 13-15: Enemy units - HP normalized, only if alive
        for i, unit in enumerate(game_state.enemy_units):
            if i < 3 and unit.is_alive():
                frame[13 + i, unit.y, unit.x] = unit.hp / self.config.max_unit_hp

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

        Reward design principles:
        1. Sparse terminal rewards for winning/losing (main objective)
        2. Dense rewards for HP changes (intermediate progress)
        3. Minimal shaping to avoid reward hacking
        4. Asymmetric penalties: losing HP is worse than dealing damage (survival first)
        5. Moderate magnitudes to avoid destabilizing Q-value estimates

        Returns:
            team_reward: shared objective reward
            unit_rewards: per-unit reward (same as team_reward, no per-unit shaping)
            done: terminal flag (all enemies dead or all my units dead)
        """
        # ---------- Count alive units ----------
        prev_enemy_alive = len(prev_game_state.enemy_alive_units)
        prev_my_alive = len(prev_game_state.my_alive_units)
        curr_enemy_alive = len(curr_game_state.enemy_alive_units)
        curr_my_alive = len(curr_game_state.my_alive_units)

        done = (curr_enemy_alive == 0) or (curr_my_alive == 0)

        # ---------- Terminal rewards ----------
        # Clear signal but not so large as to dominate early Q-estimates
        terminal = 0.0
        if curr_enemy_alive == 0 and curr_my_alive > 0:
            # Win: +3 base, +0.5 per surviving unit
            terminal = 3.0 + 0.5 * curr_my_alive
        elif curr_my_alive == 0 and curr_enemy_alive > 0:
            # Loss: -3 base, -0.5 per remaining enemy
            terminal = -3.0 - 0.5 * curr_enemy_alive
        elif curr_my_alive == 0 and curr_enemy_alive == 0:
            # Mutual destruction - slight negative (prefer winning)
            terminal = -1.0

        # ---------- HP-based dense rewards ----------
        prev_my_hp = sum(u.hp for u in prev_game_state.my_units)
        curr_my_hp = sum(u.hp for u in curr_game_state.my_units)
        prev_enemy_hp = sum(u.hp for u in prev_game_state.enemy_units)
        curr_enemy_hp = sum(u.hp for u in curr_game_state.enemy_units)

        enemy_hp_lost = max(0, prev_enemy_hp - curr_enemy_hp)
        my_hp_lost = max(0, prev_my_hp - curr_my_hp)

        # Reward for damaging enemies
        damage_reward = 0.15 * enemy_hp_lost

        # Penalty for taking damage (2x asymmetry encourages caution)
        damage_penalty = 0.3 * my_hp_lost

        # ---------- Survival bonus ----------
        # Small reward for staying alive each tick
        survival_bonus = 0.005 * curr_my_alive if not done else 0.0

        # ---------- Compute total reward ----------
        team_reward = terminal + damage_reward - damage_penalty + survival_bonus

        # ---------- Per-unit rewards ----------
        # All units share the team reward - simple and stable
        unit_rewards: dict[str, float] = {}
        for unit in curr_game_state.my_alive_units:
            unit_rewards[unit.unit_id] = team_reward

        return team_reward, unit_rewards, done
