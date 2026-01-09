"""
Strategic decision making for the AgentEx Bomberland agent.

This module handles high-level strategy:
- Game phase detection (early/mid/endgame)
- Target selection and prioritization
- Action type decisions (attack/defend/collect/explore)
- Unit coordination to avoid conflicts
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from types_ import Entity, EntityType, GameState, Point, UnitState
from danger import DangerMap, unit_in_danger
from pathfinding import find_path, find_path_to_any

# Game phase tick thresholds
EARLY_GAME_END_TICK = 100
ENDGAME_START_TICK = 200

# Strategic scoring weights
POWERUP_VALUE_BLAST = 100
POWERUP_VALUE_FREEZE = 60
ENEMY_VALUE_BASE = 80
ENEMY_VALUE_PER_MISSING_HP = 30  # Bonus for wounded enemies


class GamePhase(Enum):
    """Current phase of the game."""

    EARLY = auto()   # Focus on powerups and positioning
    MID = auto()     # Balanced offense/defense
    ENDGAME = auto() # Fire phase - survival and positioning


class ActionPriority(Enum):
    """Priority levels for different action types."""

    ESCAPE = 5       # Highest priority - survive
    DETONATE = 4     # Detonate to kill enemy
    ATTACK = 3       # Attack enemy
    COLLECT = 2      # Collect powerup
    EXPLORE = 1      # Move toward objectives
    WAIT = 0         # Do nothing


@dataclass
class Target:
    """Represents a potential target for a unit.

    Attributes:
        position: Target position on the map.
        value: Estimated value of reaching this target.
        target_type: Type of target (enemy, powerup, position).
        entity: Optional entity at the target.
        path: Computed path to target (may be None if not computed).
        path_cost: Cost of the path.
    """

    position: Point
    value: float
    target_type: str  # 'enemy', 'powerup', 'position'
    entity: Optional[Entity] = None
    unit: Optional[UnitState] = None
    path: Optional[List[Point]] = None
    path_cost: float = float('inf')


@dataclass
class UnitDecision:
    """Decision for a single unit for the current tick.

    Attributes:
        unit: The unit this decision is for.
        priority: Priority level of the chosen action.
        action_type: Type of action to perform.
        target: Target for the action.
        next_position: Position to move to (if moving).
        detonate_position: Position of bomb to detonate (if detonating).
        place_bomb: Whether to place a bomb.
    """

    unit: UnitState
    priority: ActionPriority
    action_type: str  # 'escape', 'move', 'bomb', 'detonate', 'wait'
    target: Optional[Target] = None
    next_position: Optional[Point] = None
    detonate_position: Optional[Point] = None
    place_bomb: bool = False


def get_game_phase(tick: int) -> GamePhase:
    """Determine the current game phase based on tick.

    Args:
        tick: Current game tick.

    Returns:
        Current GamePhase.
    """
    if tick < EARLY_GAME_END_TICK:
        return GamePhase.EARLY
    elif tick < ENDGAME_START_TICK:
        return GamePhase.MID
    else:
        return GamePhase.ENDGAME


def evaluate_powerup_target(
    game_state: GameState,
    unit: UnitState,
    powerup: Entity,
    danger_map: DangerMap
) -> Target:
    """Evaluate a powerup as a potential target.

    Args:
        game_state: Current game state.
        unit: Unit considering the target.
        powerup: Powerup entity.
        danger_map: Pre-computed danger map.

    Returns:
        Target object with evaluated value.
    """
    position = Point(powerup.x, powerup.y)

    # Base value by type
    if powerup.entity_type == EntityType.BLAST_POWERUP:
        base_value = POWERUP_VALUE_BLAST
    elif powerup.entity_type == EntityType.FREEZE_POWERUP:
        base_value = POWERUP_VALUE_FREEZE
    else:
        base_value = 0

    # Calculate path
    path = find_path(
        game_state,
        Point(unit.x, unit.y),
        position,
        danger_map,
        allow_destruction=False  # Don't break blocks for powerups
    )

    if path is None:
        return Target(
            position=position,
            value=0,
            target_type='powerup',
            entity=powerup,
            path=None,
            path_cost=float('inf')
        )

    path_cost = len(path)

    # Validate that the first step is actually walkable
    if len(path) > 1:
        next_pos = path[1]
        if not is_tile_walkable_now(game_state, next_pos.x, next_pos.y):
            # First step is blocked - invalid path
            return Target(
                position=position,
                value=0,
                target_type='powerup',
                entity=powerup,
                path=None,
                path_cost=float('inf')
            )

    # Check if powerup will expire before we reach it
    if powerup.expires:
        ticks_to_reach = path_cost - 1  # -1 because first element is current pos
        ticks_until_expire = powerup.expires - game_state.tick
        if ticks_to_reach >= ticks_until_expire:
            return Target(
                position=position,
                value=0,
                target_type='powerup',
                entity=powerup,
                path=path,
                path_cost=path_cost
            )

    # Adjust value by distance (closer = better)
    value = base_value / (path_cost + 1)

    # Reduce value if path goes through danger
    if danger_map.is_dangerous(powerup.x, powerup.y):
        value *= 0.5

    return Target(
        position=position,
        value=value,
        target_type='powerup',
        entity=powerup,
        path=path,
        path_cost=path_cost
    )


def is_tile_walkable_now(game_state: GameState, x: int, y: int) -> bool:
    """Check if a tile is immediately walkable (no obstacles, bombs, or units).

    Args:
        game_state: Current game state.
        x: X coordinate.
        y: Y coordinate.

    Returns:
        True if tile can be walked into right now.
    """
    # Check bounds
    if not (0 <= x < game_state.world.width and 0 <= y < game_state.world.height):
        return False

    # Check for blocking entities
    entities = game_state.entities_at(x, y)
    for entity in entities:
        if entity.entity_type in {
            EntityType.METAL_BLOCK,
            EntityType.WOOD_BLOCK,
            EntityType.ORE_BLOCK,
            EntityType.BOMB
        }:
            return False

    # Check for units
    for unit in game_state.all_units:
        if unit.is_alive() and unit.x == x and unit.y == y:
            return False

    return True


def get_blocking_entity_at(game_state: GameState, x: int, y: int) -> Optional[Entity]:
    """Get a blocking destructible entity at position, if any.

    Args:
        game_state: Current game state.
        x: X coordinate.
        y: Y coordinate.

    Returns:
        Blocking entity (wood/ore block) or None.
    """
    entities = game_state.entities_at(x, y)
    for entity in entities:
        if entity.entity_type in {EntityType.WOOD_BLOCK, EntityType.ORE_BLOCK}:
            return entity
    return None


def evaluate_enemy_target(
    game_state: GameState,
    unit: UnitState,
    enemy: UnitState,
    danger_map: DangerMap
) -> Target:
    """Evaluate an enemy as a potential target.

    Args:
        game_state: Current game state.
        unit: Unit considering the target.
        enemy: Enemy unit.
        danger_map: Pre-computed danger map.

    Returns:
        Target object with evaluated value.
    """
    position = enemy.position

    # Base value + bonus for wounded enemies
    missing_hp = 3 - enemy.hp
    base_value = ENEMY_VALUE_BASE + (ENEMY_VALUE_PER_MISSING_HP * missing_hp)

    # First try to find a path without destruction (safer, faster)
    path = find_path(
        game_state,
        Point(unit.x, unit.y),
        position,
        danger_map,
        allow_destruction=False  # Only walkable paths
    )

    if path is not None and len(path) > 1:
        next_pos = path[1]
        if is_tile_walkable_now(game_state, next_pos.x, next_pos.y):
            # Direct walkable path exists
            path_cost = len(path)
            value = base_value / (path_cost + 1)
            return Target(
                position=position,
                value=value,
                target_type='enemy',
                unit=enemy,
                path=path,
                path_cost=path_cost
            )

    # No direct walkable path - try with destruction allowed
    path_with_destruction = find_path(
        game_state,
        Point(unit.x, unit.y),
        position,
        danger_map,
        allow_destruction=True  # Allow paths through blocks
    )

    if path_with_destruction is None:
        # Completely unreachable
        return Target(
            position=position,
            value=0,
            target_type='enemy',
            unit=enemy,
            path=None,
            path_cost=float('inf')
        )

    # Path exists but may require bombing obstacles
    path_cost = len(path_with_destruction)
    # Reduce value for paths requiring destruction (more effort)
    value = base_value / (path_cost + 1) * 0.8

    return Target(
        position=position,
        value=value,
        target_type='enemy',
        unit=enemy,
        path=path_with_destruction,
        path_cost=path_cost
    )


def get_available_targets(
    game_state: GameState,
    unit: UnitState,
    danger_map: DangerMap,
    phase: GamePhase
) -> List[Target]:
    """Get all available targets for a unit.

    Args:
        game_state: Current game state.
        unit: Unit to find targets for.
        danger_map: Pre-computed danger map.
        phase: Current game phase.

    Returns:
        List of Target objects sorted by value (highest first).
    """
    targets: List[Target] = []

    # Evaluate powerups (especially important in early game)
    powerups = [
        e for e in game_state.entities
        if e.entity_type in {EntityType.BLAST_POWERUP, EntityType.FREEZE_POWERUP}
    ]

    for powerup in powerups:
        target = evaluate_powerup_target(game_state, unit, powerup, danger_map)
        if target.value > 0:
            # Boost powerup value in early game
            if phase == GamePhase.EARLY:
                target.value *= 1.5
            targets.append(target)

    # Evaluate enemies
    for enemy in game_state.enemy_alive_units:
        target = evaluate_enemy_target(game_state, unit, enemy, danger_map)
        if target.value > 0:
            # Boost enemy value in mid/endgame
            if phase in {GamePhase.MID, GamePhase.ENDGAME}:
                target.value *= 1.3
            targets.append(target)

    # Sort by value (highest first)
    targets.sort(key=lambda t: t.value, reverse=True)

    return targets


def assign_targets_to_units(
    game_state: GameState,
    units: List[UnitState],
    danger_map: DangerMap,
    phase: GamePhase
) -> Dict[str, Optional[Target]]:
    """Assign targets to units to minimize overlap and maximize value.

    Uses a greedy approach: assign highest-value targets first,
    with preference for closer units.

    Args:
        game_state: Current game state.
        units: Units to assign targets to.
        danger_map: Pre-computed danger map.
        phase: Current game phase.

    Returns:
        Dict mapping unit_id to assigned Target (or None).
    """
    assignments: Dict[str, Optional[Target]] = {u.unit_id: None for u in units}

    # Collect all (unit, target, value) tuples
    candidates: List[Tuple[UnitState, Target]] = []
    for unit in units:
        targets = get_available_targets(game_state, unit, danger_map, phase)
        for target in targets:
            candidates.append((unit, target))

    # Sort by value (highest first)
    candidates.sort(key=lambda x: x[1].value, reverse=True)

    # Track assigned targets (by position) to avoid duplicates
    assigned_positions: Set[Tuple[int, int]] = set()
    assigned_units: Set[str] = set()

    for unit, target in candidates:
        if unit.unit_id in assigned_units:
            continue

        pos_key = (target.position.x, target.position.y)

        # For enemy targets, allow multiple units to target same enemy
        if target.target_type != 'enemy' and pos_key in assigned_positions:
            continue

        assignments[unit.unit_id] = target
        assigned_units.add(unit.unit_id)
        assigned_positions.add(pos_key)

        if len(assigned_units) == len(units):
            break

    return assignments


def decide_unit_action(
    game_state: GameState,
    unit: UnitState,
    target: Optional[Target],
    danger_map: DangerMap,
    reserved_positions: Set[Tuple[int, int]],
    phase: GamePhase
) -> UnitDecision:
    """Decide what action a unit should take this tick.

    Args:
        game_state: Current game state.
        unit: Unit to decide for.
        target: Assigned target (may be None).
        danger_map: Pre-computed danger map.
        reserved_positions: Positions already claimed by other units.
        phase: Current game phase.

    Returns:
        UnitDecision for this unit.
    """
    from danger import calculate_escape_routes
    from pathfinding import find_escape_path

    current_pos = Point(unit.x, unit.y)

    # Priority 1: Escape if in danger
    if unit_in_danger(game_state, unit, danger_map):
        escape_path = find_escape_path(game_state, unit, danger_map)
        if escape_path and len(escape_path) > 1:
            next_pos = escape_path[1]
            return UnitDecision(
                unit=unit,
                priority=ActionPriority.ESCAPE,
                action_type='escape',
                next_position=next_pos
            )
        else:
            # No escape route - stay and hope for the best
            return UnitDecision(
                unit=unit,
                priority=ActionPriority.ESCAPE,
                action_type='wait'
            )

    # Priority 2: Check if we can detonate a bomb to hit an enemy
    detonation = check_detonation_opportunity(game_state, unit, danger_map)
    if detonation:
        return UnitDecision(
            unit=unit,
            priority=ActionPriority.DETONATE,
            action_type='detonate',
            detonate_position=detonation
        )

    # No target assigned - just wait
    if target is None:
        return UnitDecision(
            unit=unit,
            priority=ActionPriority.WAIT,
            action_type='wait'
        )

    # Priority 3: Move toward target or attack
    if target.path and len(target.path) > 1:
        next_pos = target.path[1]

        # Check if next position is blocked by a destructible block
        blocking_entity = get_blocking_entity_at(game_state, next_pos.x, next_pos.y)
        if blocking_entity is not None:
            # Path goes through a block - we need to bomb it first
            # Check if we should place a bomb to clear the path
            if should_place_bomb_to_clear(game_state, unit, next_pos, danger_map):
                return UnitDecision(
                    unit=unit,
                    priority=ActionPriority.ATTACK,
                    action_type='bomb',
                    target=target,
                    place_bomb=True
                )
            else:
                # Can't safely bomb - wait
                return UnitDecision(
                    unit=unit,
                    priority=ActionPriority.WAIT,
                    action_type='wait',
                    target=target
                )

        # Validate next position is actually walkable NOW
        if not is_tile_walkable_now(game_state, next_pos.x, next_pos.y):
            # Path is stale or invalid - wait
            return UnitDecision(
                unit=unit,
                priority=ActionPriority.WAIT,
                action_type='wait',
                target=target
            )

        # Check if next position is reserved
        if (next_pos.x, next_pos.y) in reserved_positions:
            # Try to find alternative
            return UnitDecision(
                unit=unit,
                priority=ActionPriority.WAIT,
                action_type='wait',
                target=target
            )

        # Check if we're adjacent to target
        if len(target.path) == 2:
            # We're one step away
            if target.target_type == 'enemy':
                # Check if we should place bomb
                if should_place_bomb(game_state, unit, target, danger_map):
                    return UnitDecision(
                        unit=unit,
                        priority=ActionPriority.ATTACK,
                        action_type='bomb',
                        target=target,
                        place_bomb=True
                    )

            # Move to target
            return UnitDecision(
                unit=unit,
                priority=ActionPriority.ATTACK if target.target_type == 'enemy' else ActionPriority.COLLECT,
                action_type='move',
                target=target,
                next_position=next_pos
            )

        # Not adjacent yet - just move
        return UnitDecision(
            unit=unit,
            priority=ActionPriority.EXPLORE,
            action_type='move',
            target=target,
            next_position=next_pos
        )

    # Target has no path - wait
    return UnitDecision(
        unit=unit,
        priority=ActionPriority.WAIT,
        action_type='wait',
        target=target
    )


def check_detonation_opportunity(
    game_state: GameState,
    unit: UnitState,
    danger_map: DangerMap
) -> Optional[Point]:
    """Check if detonating one of our bombs would hit an enemy.

    Args:
        game_state: Current game state.
        unit: Unit that placed the bombs.
        danger_map: Pre-computed danger map.

    Returns:
        Position of bomb to detonate, or None.
    """
    # Get this unit's armed bombs
    my_bombs = [
        e for e in game_state.entities
        if e.entity_type == EntityType.BOMB
        and e.owner_unit_id == unit.unit_id
        and e.is_armed(game_state.tick)
    ]

    for bomb in my_bombs:
        # Get blast tiles
        blast_tiles = game_state.get_blast_tiles_if_detonated(
            Point(bomb.x, bomb.y),
            require_armed=True
        )

        # Check if any enemy is in blast zone
        for enemy in game_state.enemy_alive_units:
            if Point(enemy.x, enemy.y) in blast_tiles:
                # Check we're not in the blast zone too
                if Point(unit.x, unit.y) not in blast_tiles:
                    return Point(bomb.x, bomb.y)

    return None


def should_place_bomb_to_clear(
    game_state: GameState,
    unit: UnitState,
    blocked_pos: Point,
    danger_map: DangerMap
) -> bool:
    """Check if we should place a bomb to clear a blocking obstacle.

    Only place a bomb if:
    1. We can place a bomb (have one available)
    2. The bomb will reach the blocking position
    3. We can safely escape after placing

    Args:
        game_state: Current game state.
        unit: Unit considering bomb placement.
        blocked_pos: Position of the blocking obstacle.
        danger_map: Pre-computed danger map.

    Returns:
        True if bomb should be placed to clear the obstacle.
    """
    from danger import can_escape_after_bomb
    from utils import can_place_bomb
    from bomb_logic import get_bomb_blast_radius, get_hypothetical_blast_tiles

    # Check if we can place a bomb
    if not can_place_bomb(game_state, unit):
        return False

    # Check if the bomb would actually reach the blocked position
    bomb_position = Point(unit.x, unit.y)
    blast_radius = get_bomb_blast_radius(unit)
    blast_tiles = get_hypothetical_blast_tiles(game_state, bomb_position, blast_radius)

    if blocked_pos not in blast_tiles:
        # Block is out of blast range - can't clear it from here
        return False

    # Check if we can escape after placing
    can_escape, escape_path = can_escape_after_bomb(game_state, unit, bomb_position)
    if not can_escape:
        return False

    # Check if escape path is safe (not dangerous)
    if escape_path:
        for point in escape_path[1:]:  # Skip current position
            if danger_map.is_dangerous(point.x, point.y):
                return False

    return True


def should_place_bomb(
    game_state: GameState,
    unit: UnitState,
    target: Target,
    danger_map: DangerMap
) -> bool:
    """Determine if a unit should place a bomb to attack target.

    Args:
        game_state: Current game state.
        unit: Unit considering bomb placement.
        target: Target being attacked.
        danger_map: Pre-computed danger map.

    Returns:
        True if bomb should be placed.
    """
    from danger import can_escape_after_bomb
    from utils import can_place_bomb

    # Check if we can place a bomb
    if not can_place_bomb(game_state, unit):
        return False

    # Check if we can escape after placing
    can_escape, escape_path = can_escape_after_bomb(game_state, unit)
    if not can_escape:
        return False

    # Check if escape path is safe
    if escape_path:
        for point in escape_path[1:]:  # Skip current position
            if danger_map.is_dangerous(point.x, point.y):
                return False

    return True


class StrategyManager:
    """Manages strategic decisions for all units.

    Coordinates unit actions to avoid conflicts and maximize effectiveness.
    """

    def __init__(self, game_state: GameState):
        """Initialize strategy manager.

        Args:
            game_state: Current game state.
        """
        self.game_state = game_state
        self.danger_map = DangerMap(game_state)
        self.phase = get_game_phase(game_state.tick)
        self.decisions: Dict[str, UnitDecision] = {}

    def compute_decisions(self) -> Dict[str, UnitDecision]:
        """Compute decisions for all units.

        Returns:
            Dict mapping unit_id to UnitDecision.
        """
        units = self.game_state.my_alive_units

        if not units:
            return {}

        # Assign targets
        target_assignments = assign_targets_to_units(
            self.game_state,
            units,
            self.danger_map,
            self.phase
        )

        # Track reserved positions for coordination
        reserved_positions: Set[Tuple[int, int]] = set()

        # Decide actions for each unit (in priority order)
        unit_decisions: List[Tuple[UnitState, UnitDecision]] = []

        for unit in units:
            target = target_assignments.get(unit.unit_id)
            decision = decide_unit_action(
                self.game_state,
                unit,
                target,
                self.danger_map,
                reserved_positions,
                self.phase
            )
            unit_decisions.append((unit, decision))

        # Sort by priority (highest first) for position reservation
        unit_decisions.sort(key=lambda x: x[1].priority.value, reverse=True)

        # Reserve positions and finalize decisions
        for unit, decision in unit_decisions:
            if decision.next_position:
                pos_key = (decision.next_position.x, decision.next_position.y)
                if pos_key in reserved_positions:
                    # Position already taken - downgrade to wait
                    decision = UnitDecision(
                        unit=unit,
                        priority=ActionPriority.WAIT,
                        action_type='wait',
                        target=decision.target
                    )
                else:
                    reserved_positions.add(pos_key)

            self.decisions[unit.unit_id] = decision

        return self.decisions
