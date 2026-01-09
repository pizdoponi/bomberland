"""
Utility functions for the AgentEx Bomberland agent.

Provides helper functions for common operations like:
- Coordinate manipulation
- Entity filtering
- Distance calculations
- Movement direction helpers
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from types_ import (
    Entity,
    EntityType,
    GameState,
    MoveAction,
    MoveDirection,
    Point,
    UnitState,
)


# Direction vectors for cardinal movements
DIRECTION_DELTAS: Dict[MoveDirection, Tuple[int, int]] = {
    MoveDirection.UP: (0, 1),
    MoveDirection.DOWN: (0, -1),
    MoveDirection.LEFT: (-1, 0),
    MoveDirection.RIGHT: (1, 0),
}

# Reverse mapping: delta to direction
DELTA_TO_DIRECTION: Dict[Tuple[int, int], MoveDirection] = {
    v: k for k, v in DIRECTION_DELTAS.items()
}


def get_neighbors(point: Point, world_width: int = 15, world_height: int = 15) -> List[Point]:
    """Get all valid neighboring points (cardinal directions only).

    Args:
        point: The center point.
        world_width: Width of the game world.
        world_height: Height of the game world.

    Returns:
        List of valid neighboring Points within world bounds.
    """
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = point.x + dx, point.y + dy
        if 0 <= nx < world_width and 0 <= ny < world_height:
            neighbors.append(Point(nx, ny))
    return neighbors


def manhattan_distance(p1: Point, p2: Point) -> int:
    """Calculate Manhattan distance between two points.

    Args:
        p1: First point.
        p2: Second point.

    Returns:
        Manhattan distance (|x1-x2| + |y1-y2|).
    """
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)


def get_direction_to_point(from_point: Point, to_point: Point) -> Optional[MoveDirection]:
    """Get the movement direction from one adjacent point to another.

    Args:
        from_point: Starting point.
        to_point: Target point (must be adjacent).

    Returns:
        MoveDirection if points are adjacent, None otherwise.
    """
    dx = to_point.x - from_point.x
    dy = to_point.y - from_point.y
    return DELTA_TO_DIRECTION.get((dx, dy))


def apply_direction(point: Point, direction: MoveDirection) -> Point:
    """Apply a movement direction to get the resulting point.

    Args:
        point: Starting point.
        direction: Direction to move.

    Returns:
        New Point after applying the direction.
    """
    dx, dy = DIRECTION_DELTAS[direction]
    return Point(point.x + dx, point.y + dy)


def get_units_by_position(game_state: GameState) -> Dict[Tuple[int, int], UnitState]:
    """Create a mapping of positions to units occupying them.

    Args:
        game_state: Current game state.

    Returns:
        Dict mapping (x, y) tuples to UnitState objects.
    """
    return {
        (unit.x, unit.y): unit
        for unit in game_state.all_units
        if unit.is_alive()
    }


def get_entities_by_position(
    game_state: GameState,
    entity_types: Optional[Set[EntityType]] = None
) -> Dict[Tuple[int, int], List[Entity]]:
    """Create a mapping of positions to entities at those positions.

    Args:
        game_state: Current game state.
        entity_types: Optional set of entity types to filter by.

    Returns:
        Dict mapping (x, y) tuples to lists of Entity objects.
    """
    result: Dict[Tuple[int, int], List[Entity]] = {}
    for entity in game_state.entities:
        if entity_types is None or entity.entity_type in entity_types:
            key = (entity.x, entity.y)
            if key not in result:
                result[key] = []
            result[key].append(entity)
    return result


def get_bombs(game_state: GameState) -> List[Entity]:
    """Get all bomb entities on the map.

    Args:
        game_state: Current game state.

    Returns:
        List of bomb Entity objects.
    """
    return [e for e in game_state.entities if e.entity_type == EntityType.BOMB]


def get_powerups(game_state: GameState) -> List[Entity]:
    """Get all powerup entities on the map.

    Args:
        game_state: Current game state.

    Returns:
        List of powerup Entity objects (blast and freeze powerups).
    """
    return [
        e for e in game_state.entities
        if e.entity_type in {EntityType.BLAST_POWERUP, EntityType.FREEZE_POWERUP}
    ]


def get_destructible_blocks(game_state: GameState) -> List[Entity]:
    """Get all destructible block entities (wood and ore).

    Args:
        game_state: Current game state.

    Returns:
        List of destructible block Entity objects.
    """
    return [
        e for e in game_state.entities
        if e.entity_type in {EntityType.WOOD_BLOCK, EntityType.ORE_BLOCK}
    ]


def is_position_blocked(
    game_state: GameState,
    x: int,
    y: int,
    ignore_units: bool = False,
    ignore_bombs: bool = False
) -> bool:
    """Check if a position is blocked by solid entities or units.

    Args:
        game_state: Current game state.
        x: X coordinate to check.
        y: Y coordinate to check.
        ignore_units: If True, ignore units when checking.
        ignore_bombs: If True, ignore bombs when checking.

    Returns:
        True if position is blocked, False otherwise.
    """
    # Check world bounds
    if not (0 <= x < game_state.world.width and 0 <= y < game_state.world.height):
        return True

    # Check for units
    if not ignore_units:
        for unit in game_state.all_units:
            if unit.is_alive() and unit.x == x and unit.y == y:
                return True

    # Check for blocking entities
    blocking_types = {EntityType.METAL_BLOCK, EntityType.ORE_BLOCK, EntityType.WOOD_BLOCK}
    if not ignore_bombs:
        blocking_types.add(EntityType.BOMB)

    entities = game_state.entities_at(x, y)
    return any(e.entity_type in blocking_types for e in entities)


def count_escape_routes(
    game_state: GameState,
    point: Point,
    blocked_positions: Optional[Set[Tuple[int, int]]] = None
) -> int:
    """Count the number of unblocked adjacent tiles.

    Useful for evaluating how trapped a position is.

    Args:
        game_state: Current game state.
        point: Position to evaluate.
        blocked_positions: Additional positions to consider blocked.

    Returns:
        Number of walkable adjacent tiles (0-4).
    """
    blocked = blocked_positions or set()
    count = 0

    for neighbor in get_neighbors(point, game_state.world.width, game_state.world.height):
        if (neighbor.x, neighbor.y) in blocked:
            continue
        if not is_position_blocked(game_state, neighbor.x, neighbor.y, ignore_units=True):
            count += 1

    return count


def get_agent_bomb_count(game_state: GameState, agent_id: str) -> int:
    """Get the number of bombs currently placed by an agent.

    Args:
        game_state: Current game state.
        agent_id: Agent ID ('a' or 'b').

    Returns:
        Number of bombs placed by the agent's units.
    """
    agent_unit_ids = set()
    for unit in game_state.all_units:
        if unit.agent_id.value == agent_id:
            agent_unit_ids.add(unit.unit_id)

    return sum(
        1 for e in game_state.entities
        if e.entity_type == EntityType.BOMB and e.owner_unit_id in agent_unit_ids
    )


def can_place_bomb(game_state: GameState, unit: UnitState) -> bool:
    """Check if a unit can place a bomb.

    Args:
        game_state: Current game state.
        unit: The unit to check.

    Returns:
        True if the unit can place a bomb, False otherwise.
    """
    # Check agent bomb limit (max 3 per agent)
    agent_bomb_count = get_agent_bomb_count(game_state, unit.agent_id.value)
    if agent_bomb_count >= 3:
        return False

    # Check if there's already a bomb at unit's position
    entities_here = game_state.entities_at(unit.x, unit.y)
    if any(e.entity_type == EntityType.BOMB for e in entities_here):
        return False

    return True


def get_total_hp(units: List[UnitState]) -> int:
    """Get total HP across a list of units.

    Args:
        units: List of units.

    Returns:
        Total HP of all units.
    """
    return sum(max(0, unit.hp) for unit in units)


def get_alive_count(units: List[UnitState]) -> int:
    """Get count of alive units.

    Args:
        units: List of units.

    Returns:
        Number of units with HP > 0.
    """
    return sum(1 for unit in units if unit.is_alive())


def path_to_moves(unit_id: str, path: List[Point]) -> List[MoveAction]:
    """Convert a path of points to a list of move actions.

    Args:
        unit_id: ID of the unit that will move.
        path: List of points representing the path (including start).

    Returns:
        List of MoveAction objects (one fewer than path length).
    """
    if len(path) < 2:
        return []

    moves = []
    for i in range(len(path) - 1):
        direction = get_direction_to_point(path[i], path[i + 1])
        if direction:
            moves.append(MoveAction(unit_id=unit_id, move=direction))

    return moves
