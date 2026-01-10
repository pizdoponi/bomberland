"""
Bomb placement and detonation logic for the AgentEx Bomberland agent.

This module handles:
- Deciding when and where to place bombs
- Retreat path planning after bomb placement
- Detonation timing and opportunity detection
- Chain detonation analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from types_ import (
    ActionPacket,
    BombAction,
    DetonateAction,
    Entity,
    EntityType,
    GameState,
    MoveAction,
    Point,
    SkipAction,
    UnitState,
)
from danger import DangerMap, can_escape_after_bomb
from pathfinding import find_path
from utils import can_place_bomb, get_neighbors, manhattan_distance


# Engine constants
BOMB_ARMED_TICKS = 5
BLAST_DURATION_TICKS = 5
MAX_BOMBS_PER_AGENT = 3


@dataclass
class BombPlan:
    """Plan for placing and detonating a bomb.

    Attributes:
        bomb_position: Where to place the bomb.
        retreat_path: Path to retreat after placing.
        detonate_after: Number of ticks after placement to detonate.
        expected_hits: Units expected to be hit.
        can_escape: Whether unit can escape after placing.
    """

    bomb_position: Point
    retreat_path: List[Point]
    detonate_after: int
    expected_hits: List[UnitState]
    can_escape: bool


@dataclass
class ActionSequence:
    """Sequence of actions for a unit to execute over multiple ticks.

    Attributes:
        unit_id: ID of the unit.
        actions: List of actions to execute in order.
        current_index: Index of next action to execute.
    """

    unit_id: str
    actions: List[ActionPacket]
    current_index: int = 0

    def get_next_action(self) -> Optional[ActionPacket]:
        """Get the next action in the sequence."""
        if self.current_index < len(self.actions):
            action = self.actions[self.current_index]
            self.current_index += 1
            return action
        return None

    def is_complete(self) -> bool:
        """Check if sequence is complete."""
        return self.current_index >= len(self.actions)


def get_bomb_blast_radius(unit: UnitState) -> int:
    """Get the blast radius for bombs placed by a unit.

    Args:
        unit: Unit that would place the bomb.

    Returns:
        Blast radius in tiles.
    """
    diameter = unit.blast_diameter if unit.blast_diameter else 3
    return max(0, (diameter - 1) // 2)


def get_hypothetical_blast_tiles(
    game_state: GameState,
    bomb_position: Point,
    blast_radius: int
) -> Set[Point]:
    """Get tiles that would be affected by a bomb at the given position.

    Args:
        game_state: Current game state.
        bomb_position: Where the bomb would be placed.
        blast_radius: Blast radius in tiles.

    Returns:
        Set of Points in the blast zone.
    """
    tiles = {bomb_position}
    width = game_state.world.width
    height = game_state.world.height

    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for dist in range(1, blast_radius + 1):
            nx, ny = bomb_position.x + dx * dist, bomb_position.y + dy * dist

            if not (0 <= nx < width and 0 <= ny < height):
                break

            tiles.add(Point(nx, ny))

            # Check for blocking entities
            entities_here = game_state.entities_at(nx, ny)
            if any(e.entity_type in {
                EntityType.METAL_BLOCK,
                EntityType.ORE_BLOCK,
                EntityType.WOOD_BLOCK
            } for e in entities_here):
                break

    return tiles


def find_retreat_path(
    game_state: GameState,
    unit: UnitState,
    bomb_position: Point,
    danger_map: Optional[DangerMap] = None
) -> Optional[List[Point]]:
    """Find a safe retreat path after placing a bomb.

    Args:
        game_state: Current game state.
        unit: Unit placing the bomb.
        bomb_position: Where the bomb will be placed.
        danger_map: Pre-computed danger map.

    Returns:
        Retreat path, or None if no safe retreat.
    """
    blast_radius = get_bomb_blast_radius(unit)
    blast_tiles = get_hypothetical_blast_tiles(game_state, bomb_position, blast_radius)

    # Find safe tiles outside blast radius
    width = game_state.world.width
    height = game_state.world.height

    safe_targets: List[Point] = []
    min_distance = blast_radius + 1

    for x in range(width):
        for y in range(height):
            point = Point(x, y)
            if point in blast_tiles:
                continue

            # Check if walkable
            if not game_state.is_walkable(x, y, ignore_units=True, ignore_bombs=True):
                continue

            # Check if reachable (within reasonable distance)
            dist = manhattan_distance(Point(unit.x, unit.y), point)
            if dist > blast_radius + 3:  # Don't go too far
                continue

            safe_targets.append(point)

    if not safe_targets:
        return None

    # Find shortest path to any safe target
    best_path = None
    best_length = float('inf')

    for target in safe_targets:
        # Use pathfinding that avoids the blast zone
        path = find_path(
            game_state,
            Point(unit.x, unit.y),
            target,
            danger_map,
            allow_destruction=False,
            avoid_danger=False,  # We're already considering blast zone
            excluded_positions={(bomb_position.x, bomb_position.y)}
        )

        if path and len(path) < best_length:
            # Verify path doesn't go through blast zone (except start)
            valid = True
            for i, p in enumerate(path[1:], 1):
                if p in blast_tiles and i < len(path) - 1:
                    # Allow passing through if we can exit before detonation
                    if i > BOMB_ARMED_TICKS:
                        valid = False
                        break

            if valid:
                best_path = path
                best_length = len(path)

    return best_path


def create_bomb_and_retreat_sequence(
    game_state: GameState,
    unit: UnitState,
    bomb_position: Optional[Point] = None,
    danger_map: Optional[DangerMap] = None
) -> Optional[ActionSequence]:
    """Create an action sequence for placing bomb, retreating, and detonating.

    Args:
        game_state: Current game state.
        unit: Unit to create sequence for.
        bomb_position: Where to place bomb (default: unit position).
        danger_map: Pre-computed danger map.

    Returns:
        ActionSequence, or None if cannot safely execute.
    """
    if bomb_position is None:
        bomb_position = Point(unit.x, unit.y)

    # Check if we can place a bomb
    if not can_place_bomb(game_state, unit):
        return None

    # Find retreat path
    retreat_path = find_retreat_path(game_state, unit, bomb_position, danger_map)
    if retreat_path is None or len(retreat_path) < 2:
        return None

    actions: List[ActionPacket] = []

    # Place bomb
    actions.append(BombAction(unit_id=unit.unit_id))

    # Retreat moves
    for i in range(len(retreat_path) - 1):
        move = MoveAction.from_points(unit.unit_id, retreat_path[i], retreat_path[i + 1])
        if move:
            actions.append(move)

    # Wait for bomb to arm (if needed)
    retreat_ticks = len(retreat_path) - 1
    wait_ticks = max(0, BOMB_ARMED_TICKS - retreat_ticks)

    for _ in range(wait_ticks):
        actions.append(SkipAction(unit_id=unit.unit_id))

    # Detonate
    actions.append(DetonateAction(unit_id=unit.unit_id, target=bomb_position))

    # Wait for blast to clear
    for _ in range(BLAST_DURATION_TICKS):
        actions.append(SkipAction(unit_id=unit.unit_id))

    return ActionSequence(unit_id=unit.unit_id, actions=actions)


def evaluate_bomb_placement(
    game_state: GameState,
    unit: UnitState,
    danger_map: Optional[DangerMap] = None
) -> Optional[BombPlan]:
    """Evaluate if placing a bomb at current position is worthwhile.

    Args:
        game_state: Current game state.
        unit: Unit considering bomb placement.
        danger_map: Pre-computed danger map.

    Returns:
        BombPlan if placement is good, None otherwise.
    """
    bomb_position = Point(unit.x, unit.y)

    # Check if we can place
    if not can_place_bomb(game_state, unit):
        return None

    # Create danger map if not provided
    if danger_map is None:
        danger_map = DangerMap(game_state)

    # Check escape (pass danger_map to check existing bombs)
    can_escape, escape_path = can_escape_after_bomb(game_state, unit, bomb_position, danger_map=danger_map)
    if not can_escape or escape_path is None:
        return None

    # Calculate blast tiles
    blast_radius = get_bomb_blast_radius(unit)
    blast_tiles = get_hypothetical_blast_tiles(game_state, bomb_position, blast_radius)

    # Check if any friendly units would be hit (excluding self)
    for friendly in game_state.my_alive_units:
        if friendly.unit_id == unit.unit_id:
            continue  # We already checked our escape
        friendly_pos = Point(friendly.x, friendly.y)
        if friendly_pos in blast_tiles:
            # Don't place bomb if it would hit a friendly
            return None

    # Check what we'd hit
    enemies_hit: List[UnitState] = []
    for enemy in game_state.enemy_alive_units:
        if Point(enemy.x, enemy.y) in blast_tiles:
            enemies_hit.append(enemy)

    blocks_hit: List[Entity] = []
    for tile in blast_tiles:
        entities = game_state.entities_at(tile.x, tile.y)
        for e in entities:
            if e.entity_type in {EntityType.WOOD_BLOCK, EntityType.ORE_BLOCK}:
                blocks_hit.append(e)

    # Not worth it if nothing to hit
    if not enemies_hit and not blocks_hit:
        return None

    return BombPlan(
        bomb_position=bomb_position,
        retreat_path=escape_path,
        detonate_after=BOMB_ARMED_TICKS,
        expected_hits=enemies_hit,
        can_escape=True
    )


def check_immediate_detonation(
    game_state: GameState,
    unit: UnitState
) -> Optional[Point]:
    """Check if we should immediately detonate one of our bombs.

    Conditions for immediate detonation:
    - Bomb is armed
    - Enemy is in blast zone
    - We are not in blast zone (or invulnerable)
    - No friendly units are in blast zone (or invulnerable)

    Args:
        game_state: Current game state.
        unit: Unit that placed bombs.

    Returns:
        Position of bomb to detonate, or None.
    """
    # Get unit's armed bombs
    my_bombs = [
        e for e in game_state.entities
        if e.entity_type == EntityType.BOMB
        and e.owner_unit_id == unit.unit_id
        and e.is_armed(game_state.tick)
    ]

    for bomb in my_bombs:
        blast_tiles = game_state.get_blast_tiles_if_detonated(
            Point(bomb.x, bomb.y),
            require_armed=True
        )

        # Check if any enemy is in blast
        enemy_in_blast = False
        for enemy in game_state.enemy_alive_units:
            if Point(enemy.x, enemy.y) in blast_tiles:
                # Check enemy isn't invulnerable
                if not enemy.is_invulnerable(game_state.tick):
                    enemy_in_blast = True
                    break

        if not enemy_in_blast:
            continue

        # Check if we're safe
        unit_pos = Point(unit.x, unit.y)
        if unit_pos in blast_tiles:
            # Check if invulnerable
            if not unit.is_invulnerable(game_state.tick):
                continue

        # Check if any friendly units would be hit
        friendly_in_blast = False
        for friendly in game_state.my_alive_units:
            if friendly.unit_id == unit.unit_id:
                continue  # Already checked ourselves above
            friendly_pos = Point(friendly.x, friendly.y)
            if friendly_pos in blast_tiles:
                if not friendly.is_invulnerable(game_state.tick):
                    friendly_in_blast = True
                    break

        if friendly_in_blast:
            continue  # Don't detonate if it would hit a friendly

        return Point(bomb.x, bomb.y)

    return None


def get_optimal_bomb_positions(
    game_state: GameState,
    unit: UnitState,
    target_enemy: UnitState,
    danger_map: Optional[DangerMap] = None
) -> List[Point]:
    """Find optimal positions to place a bomb to hit a target enemy.

    Args:
        game_state: Current game state.
        unit: Unit that would place bomb.
        target_enemy: Enemy to target.
        danger_map: Pre-computed danger map.

    Returns:
        List of Points where placing a bomb could hit the enemy.
    """
    blast_radius = get_bomb_blast_radius(unit)
    enemy_pos = Point(target_enemy.x, target_enemy.y)

    optimal_positions: List[Point] = []

    # Check positions in blast range of enemy
    for dx in range(-blast_radius, blast_radius + 1):
        for dy in range(-blast_radius, blast_radius + 1):
            # Only cardinal directions
            if dx != 0 and dy != 0:
                continue

            x, y = enemy_pos.x + dx, enemy_pos.y + dy

            if not (0 <= x < game_state.world.width and 0 <= y < game_state.world.height):
                continue

            pos = Point(x, y)

            # Check if we could place bomb here
            if not game_state.is_walkable(x, y, ignore_units=True, ignore_bombs=True):
                continue

            # Check if this position would actually hit the enemy
            hypothetical_blast = get_hypothetical_blast_tiles(game_state, pos, blast_radius)
            if enemy_pos not in hypothetical_blast:
                continue

            # Check if we can escape from this position (pass danger_map)
            can_escape, _ = can_escape_after_bomb(game_state, unit, pos, danger_map=danger_map)
            if not can_escape:
                continue

            optimal_positions.append(pos)

    # Sort by distance to unit (closer is better)
    optimal_positions.sort(key=lambda p: manhattan_distance(Point(unit.x, unit.y), p))

    return optimal_positions


class BombManager:
    """Manages bomb-related decisions and action sequences.

    Tracks ongoing bomb sequences and coordinates bomb usage across units.
    """

    def __init__(self):
        """Initialize bomb manager."""
        self.active_sequences: Dict[str, ActionSequence] = {}
        self.pending_bombs: Dict[str, Point] = {}  # unit_id -> bomb position

    def has_active_sequence(self, unit_id: str) -> bool:
        """Check if a unit has an active bomb sequence."""
        return unit_id in self.active_sequences and not self.active_sequences[unit_id].is_complete()

    def get_next_action(self, unit_id: str) -> Optional[ActionPacket]:
        """Get next action from active sequence for a unit."""
        if unit_id in self.active_sequences:
            return self.active_sequences[unit_id].get_next_action()
        return None

    def start_bomb_sequence(
        self,
        game_state: GameState,
        unit: UnitState,
        danger_map: Optional[DangerMap] = None
    ) -> bool:
        """Start a bomb and retreat sequence for a unit.

        Args:
            game_state: Current game state.
            unit: Unit to start sequence for.
            danger_map: Pre-computed danger map.

        Returns:
            True if sequence was started, False otherwise.
        """
        sequence = create_bomb_and_retreat_sequence(game_state, unit, danger_map=danger_map)
        if sequence:
            self.active_sequences[unit.unit_id] = sequence
            self.pending_bombs[unit.unit_id] = Point(unit.x, unit.y)
            return True
        return False

    def clear_completed_sequences(self) -> None:
        """Remove completed sequences from tracking."""
        to_remove = [
            uid for uid, seq in self.active_sequences.items()
            if seq.is_complete()
        ]
        for uid in to_remove:
            del self.active_sequences[uid]
            if uid in self.pending_bombs:
                del self.pending_bombs[uid]

    def reset_unit_sequence(self, unit_id: str) -> None:
        """Reset/cancel sequence for a unit (e.g., if situation changed)."""
        if unit_id in self.active_sequences:
            del self.active_sequences[unit_id]
        if unit_id in self.pending_bombs:
            del self.pending_bombs[unit_id]
