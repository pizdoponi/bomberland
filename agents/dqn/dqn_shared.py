from __future__ import annotations

from collections import deque
from typing import Deque, List, Set

import numpy as np
from dqn_config import DQNConfig
from dqn_model import ActionType
from types_ import Entity, EntityType, Point
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
        _actions_taken: dict[str, ActionType],
    ) -> tuple[float, dict[str, float], bool]:
        """
        Compute rewards for team and individual units.

        Reward design principles:
        1. TERMINAL-FOCUSED: Win/lose dominates (+1/-0.9)
        2. INDIVIDUAL ATTRIBUTION: Units get credit for their own damage/kills
        3. BOUNDED [-1, 1]: All rewards clipped to this range
        4. SURVIVAL > AGGRESSION: Staying alive is more important than dealing damage
        5. DENSE FEEDBACK: Small rewards for intermediate progress

        Reward components (per unit):
        - Terminal: +1.0 win, -0.9 lose, -0.3 draw (shared)
        - Enemy damage dealt: +0.15/HP (attributed to attacker whose bomb detonated this tick)
        - Damage taken: -0.12/HP (victim, any source - teaches avoidance)
        - Friendly fire caused: -0.18/HP (attacker whose bomb detonated this tick hit ally)
        - Obstacle destruction: +0.01/block (encourages path creation)
        - Tick penalty: -0.0005/tick (encourages finishing games, very small)

        Returns:
            team_reward: shared objective reward (clipped to [-1, 1])
            unit_rewards: per-unit reward with individual attribution (clipped to [-1, 1])
            done: terminal flag
        """
        # ---------- Count alive units ----------
        curr_enemy_alive = len(curr_game_state.enemy_alive_units)
        curr_my_alive = len(curr_game_state.my_alive_units)

        done = (curr_enemy_alive == 0) or (curr_my_alive == 0)

        # ---------- Terminal rewards (primary signal) ----------
        terminal = 0.0
        if done:
            if curr_enemy_alive == 0 and curr_my_alive > 0:
                terminal = 1.0  # Won
            elif curr_my_alive == 0 and curr_enemy_alive > 0:
                terminal = -0.9  # Lost
            else:
                terminal = -0.3  # Draw

        # ---------- Team HP differential (small, shared) ----------
        prev_enemy_hp = sum(u.hp for u in prev_game_state.enemy_units)
        curr_enemy_hp = sum(u.hp for u in curr_game_state.enemy_units)
        prev_my_hp = sum(u.hp for u in prev_game_state.my_units)
        curr_my_hp = sum(u.hp for u in curr_game_state.my_units)

        enemy_hp_lost = max(0, prev_enemy_hp - curr_enemy_hp)
        my_hp_lost = max(0, prev_my_hp - curr_my_hp)

        # Team-level HP shaping
        team_hp_reward = 0.03 * enemy_hp_lost - 0.025 * my_hp_lost

        # ---------- Team reward ----------
        team_reward = terminal + team_hp_reward
        team_reward = max(-1.0, min(1.0, team_reward))

        # ---------- Compute blast ownership for damage attribution ----------
        # Map each tile with active blast to the unit(s) whose bomb caused it THIS TICK
        # Only attribute if bomb existed in prev_state but detonated (not in curr_state)
        blast_ownership: dict[Point, set[str]] = {}

        # Get bomb positions that exist in current state (not yet detonated)
        curr_bomb_positions = {bomb.position for bomb in curr_game_state._all_bombs or []}

        # Track which bombs detonated this tick for obstacle attribution
        detonated_bombs: List[Entity] = []

        # Check bombs from previous state that are no longer present (detonated this tick)
        for bomb in prev_game_state._all_bombs or []:
            owner = bomb.owner_unit_id
            if owner is None:
                continue
            # Only attribute if this bomb detonated this tick (was in prev, not in curr)
            if bomb.position in curr_bomb_positions:
                continue  # Bomb still exists, didn't detonate

            detonated_bombs.append(bomb)

            # This bomb detonated - attribute its blast tiles to the owner
            blast_tiles = prev_game_state.get_blast_tiles_if_detonated(
                bomb.position, require_armed=False
            )
            for blast_pos in blast_tiles:
                if blast_pos not in blast_ownership:
                    blast_ownership[blast_pos] = set()
                blast_ownership[blast_pos].add(owner)

        # ---------- Compute per-unit damage dealt/taken ----------
        my_unit_ids = {u.unit_id for u in curr_game_state.my_units}
        # Also include units that may have just died
        all_my_unit_ids = my_unit_ids | {u.unit_id for u in prev_game_state.my_units}

        # Track damage dealt to enemies by each of my units (positive)
        damage_dealt_by_unit: dict[str, float] = {uid: 0.0 for uid in all_my_unit_ids}

        # Track friendly fire caused by each of my units (damage to allies from our bombs this tick)
        friendly_fire_by_unit: dict[str, float] = {uid: 0.0 for uid in all_my_unit_ids}

        # Track HP lost by each of my units (any source)
        hp_lost_by_unit: dict[str, int] = {uid: 0 for uid in all_my_unit_ids}

        # Track obstacles destroyed by each of my units
        obstacles_destroyed_by_unit: dict[str, float] = {uid: 0.0 for uid in all_my_unit_ids}

        # Calculate HP lost per unit (any source: own bomb, ally bomb, enemy bomb, walking into blast, fire)
        for ally in prev_game_state.my_units:
            curr_ally = curr_game_state.get_unit(ally.unit_id)
            curr_hp = curr_ally.hp if curr_ally else 0
            hp_lost_by_unit[ally.unit_id] = max(0, ally.hp - curr_hp)

        # Calculate damage dealt to enemies and attribute to my units
        for enemy in prev_game_state.enemy_units:
            curr_enemy = curr_game_state.get_unit(enemy.unit_id)
            curr_hp = curr_enemy.hp if curr_enemy else 0
            hp_lost = max(0, enemy.hp - curr_hp)

            if hp_lost > 0:
                enemy_pos = Point(enemy.x, enemy.y)
                # Check if enemy was hit by a blast we own (this tick)
                if enemy_pos in blast_ownership:
                    my_units_responsible = blast_ownership[enemy_pos] & all_my_unit_ids
                    if my_units_responsible:
                        # Split credit among responsible units
                        credit_per_unit = hp_lost / len(my_units_responsible)
                        for uid in my_units_responsible:
                            damage_dealt_by_unit[uid] += credit_per_unit

        # Calculate friendly fire: damage to allies caused by my units' bombs THIS TICK
        # Only attribute if the bomb detonated this tick (not if ally walked into existing blast)
        for ally in prev_game_state.my_units:
            hp_lost = hp_lost_by_unit[ally.unit_id]

            if hp_lost > 0:
                ally_pos = Point(ally.x, ally.y)
                # Check if ally was hit by a blast from one of our bombs that detonated this tick
                if ally_pos in blast_ownership:
                    my_units_responsible = blast_ownership[ally_pos] & all_my_unit_ids
                    if my_units_responsible:
                        # Penalize the unit(s) whose bomb caused the friendly fire
                        penalty_per_unit = hp_lost / len(my_units_responsible)
                        for uid in my_units_responsible:
                            friendly_fire_by_unit[uid] += penalty_per_unit

        # Calculate obstacles destroyed by my units' bombs this tick
        # Count wood and ore blocks that were in prev_state but not in curr_state
        prev_blocks: dict[Point, EntityType] = {}
        for entity in prev_game_state.entities:
            if entity.entity_type in {EntityType.WOOD_BLOCK, EntityType.ORE_BLOCK}:
                prev_blocks[entity.position] = entity.entity_type

        curr_block_positions: Set[Point] = set()
        for entity in curr_game_state.entities:
            if entity.entity_type in {EntityType.WOOD_BLOCK, EntityType.ORE_BLOCK}:
                curr_block_positions.add(entity.position)

        # Find blocks that were destroyed
        for block_pos, _block_type in prev_blocks.items():
            if block_pos not in curr_block_positions:
                # Block was destroyed - check if our bomb did it
                if block_pos in blast_ownership:
                    my_units_responsible = blast_ownership[block_pos] & all_my_unit_ids
                    if my_units_responsible:
                        credit_per_unit = 1.0 / len(my_units_responsible)
                        for uid in my_units_responsible:
                            obstacles_destroyed_by_unit[uid] += credit_per_unit

        # ---------- Per-unit rewards with individual attribution ----------
        unit_rewards: dict[str, float] = {}

        # Very small tick penalty to encourage finishing games (shared by alive units)
        tick_penalty = -0.0005

        for unit in curr_game_state.my_units:
            unit_id = unit.unit_id

            # Start with terminal reward (shared goal)
            unit_reward = terminal  # +1.0 win, -0.9 lose, -0.3 draw

            # Tick penalty (encourages finishing games, very minor)
            unit_reward += tick_penalty

            # Individual damage dealt to enemies: +0.15 per HP
            # (max 0.45 for a kill, 1.35 for killing all 3 enemies - will be clipped)
            individual_damage_dealt = damage_dealt_by_unit.get(unit_id, 0.0)
            unit_reward += 0.15 * individual_damage_dealt

            # Damage taken penalty: -0.12 per HP this unit lost (any source)
            # Teaches unit to avoid danger zones, not walk into blasts
            individual_hp_lost = hp_lost_by_unit.get(unit_id, 0)
            unit_reward -= 0.12 * individual_hp_lost

            # Friendly fire penalty: -0.18 per HP this unit's bomb caused to allies
            # Only counts if our bomb detonated this tick (not ally walking into old blast)
            individual_friendly_fire = friendly_fire_by_unit.get(unit_id, 0.0)
            unit_reward -= 0.18 * individual_friendly_fire

            # Obstacle destruction bonus: +0.01 per block destroyed
            # Encourages path creation to reach enemies
            individual_obstacles = obstacles_destroyed_by_unit.get(unit_id, 0.0)
            unit_reward += 0.01 * individual_obstacles

            # Clip to [-1, 1]
            unit_rewards[unit_id] = max(-1.0, min(1.0, unit_reward))

        # Handle units that died this tick
        for unit in prev_game_state.my_alive_units:
            if unit.unit_id not in unit_rewards:
                unit_id = unit.unit_id
                # Unit died - start with terminal
                death_reward = terminal
                # Add tick penalty
                death_reward += tick_penalty
                # Add any damage this unit dealt before dying
                death_reward += 0.15 * damage_dealt_by_unit.get(unit_id, 0.0)
                # Subtract HP lost (includes death HP)
                death_reward -= 0.12 * hp_lost_by_unit.get(unit_id, 0)
                # Subtract any friendly fire this unit caused
                death_reward -= 0.18 * friendly_fire_by_unit.get(unit_id, 0.0)
                # Add obstacle destruction
                death_reward += 0.01 * obstacles_destroyed_by_unit.get(unit_id, 0.0)
                unit_rewards[unit_id] = max(-1.0, min(1.0, death_reward))

        return team_reward, unit_rewards, done
