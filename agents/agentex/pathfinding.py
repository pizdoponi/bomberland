"""
Danger-aware pathfinding for the AgentEx Bomberland agent.

Implements A* pathfinding with costs that account for:
- Distance traveled
- Danger zones (bombs, blasts, fire)
- Destructible blocks (with destruction cost)
- Unit positions
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from types_ import EntityType, GameState, Point, UnitState
from danger import DangerMap


# Cost constants for pathfinding
COST_MOVE_EMPTY = 1
COST_DESTROY_PER_HP = 6  # (bomb + retreat + detonate + return) * hp
COST_DANGER_ARMED = 100
COST_DANGER_UNARMED = 50
COST_DANGER_ACTIVE = float('inf')
COST_BLOCKED = float('inf')


@dataclass(order=True)
class PathNode:
    """Node in the pathfinding priority queue.

    Attributes:
        f_cost: Total estimated cost (g_cost + h_cost).
        g_cost: Cost from start to this node.
        position: Grid position.
        path: Path from start to this node.
    """

    f_cost: float
    g_cost: float = field(compare=False)
    position: Tuple[int, int] = field(compare=False)
    path: List[Point] = field(compare=False, default_factory=list)


def heuristic(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """Manhattan distance heuristic for A*.

    Args:
        pos: Current position.
        goal: Goal position.

    Returns:
        Estimated distance to goal.
    """
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def find_path(
    game_state: GameState,
    start: Point,
    goal: Point,
    danger_map: Optional[DangerMap] = None,
    avoid_units: bool = True,
    avoid_danger: bool = True,
    allow_destruction: bool = True,
    max_cost: float = float('inf'),
    excluded_positions: Optional[Set[Tuple[int, int]]] = None
) -> Optional[List[Point]]:
    """Find a path from start to goal using danger-aware A*.

    Args:
        game_state: Current game state.
        start: Starting position.
        goal: Goal position.
        danger_map: Pre-computed danger map (creates one if not provided).
        avoid_units: If True, avoid positions occupied by other units.
        avoid_danger: If True, penalize/avoid danger zones.
        allow_destruction: If True, allow paths through destructible blocks.
        max_cost: Maximum acceptable path cost.
        excluded_positions: Additional positions to avoid.

    Returns:
        List of Points from start to goal (inclusive), or None if no path found.
    """
    if danger_map is None:
        danger_map = DangerMap(game_state)

    start_key = (start.x, start.y)
    goal_key = (goal.x, goal.y)
    excluded = excluded_positions or set()

    # Early exit if at goal
    if start_key == goal_key:
        return [start]

    width = game_state.world.width
    height = game_state.world.height

    # Get unit positions (excluding start position's unit)
    unit_positions: Set[Tuple[int, int]] = set()
    if avoid_units:
        for unit in game_state.all_units:
            if unit.is_alive() and (unit.x, unit.y) != start_key:
                unit_positions.add((unit.x, unit.y))

    def get_tile_cost(x: int, y: int) -> float:
        """Calculate cost to enter a tile."""
        # Check bounds
        if not (0 <= x < width and 0 <= y < height):
            return COST_BLOCKED

        # Check excluded positions
        if (x, y) in excluded:
            return COST_BLOCKED

        # Check unit positions (but allow goal position)
        if (x, y) in unit_positions and (x, y) != goal_key:
            return COST_BLOCKED

        # Check entities
        entities = game_state.entities_at(x, y)

        # Metal blocks are always impassable
        if any(e.entity_type == EntityType.METAL_BLOCK for e in entities):
            return COST_BLOCKED

        # Calculate base cost
        cost = COST_MOVE_EMPTY

        # Add destruction cost for destructible blocks
        destruction_hp = 0
        for entity in entities:
            if entity.entity_type == EntityType.WOOD_BLOCK:
                destruction_hp += entity.hp if entity.hp else 1
            elif entity.entity_type == EntityType.ORE_BLOCK:
                destruction_hp += entity.hp if entity.hp else 3

        if destruction_hp > 0:
            if not allow_destruction:
                return COST_BLOCKED
            cost += COST_DESTROY_PER_HP * destruction_hp

        # Check for bombs (solid obstacles unless we're allowing destruction)
        if any(e.entity_type == EntityType.BOMB for e in entities):
            return COST_BLOCKED

        # Add danger cost
        if avoid_danger:
            danger_level = danger_map.get_danger_level(x, y)
            if danger_level >= 1000:
                return COST_BLOCKED  # Active blast or fire
            cost += danger_level

        return cost

    # A* search
    pq: List[PathNode] = []
    start_node = PathNode(
        f_cost=heuristic(start_key, goal_key),
        g_cost=0,
        position=start_key,
        path=[start]
    )
    heapq.heappush(pq, start_node)

    visited: Dict[Tuple[int, int], float] = {start_key: 0}

    while pq:
        node = heapq.heappop(pq)
        cx, cy = node.position

        # Skip if we've found a better path to this node
        if node.g_cost > visited.get(node.position, float('inf')):
            continue

        # Goal reached
        if node.position == goal_key:
            return node.path

        # Explore neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            neighbor_key = (nx, ny)

            tile_cost = get_tile_cost(nx, ny)
            if tile_cost >= COST_BLOCKED:
                continue

            new_g_cost = node.g_cost + tile_cost

            # Skip if exceeds max cost
            if new_g_cost > max_cost:
                continue

            # Skip if we've found a better path
            if new_g_cost >= visited.get(neighbor_key, float('inf')):
                continue

            visited[neighbor_key] = new_g_cost
            new_f_cost = new_g_cost + heuristic(neighbor_key, goal_key)
            new_path = node.path + [Point(nx, ny)]

            heapq.heappush(pq, PathNode(
                f_cost=new_f_cost,
                g_cost=new_g_cost,
                position=neighbor_key,
                path=new_path
            ))

    return None


def find_path_to_any(
    game_state: GameState,
    start: Point,
    goals: List[Point],
    danger_map: Optional[DangerMap] = None,
    **kwargs
) -> Tuple[Optional[List[Point]], Optional[Point]]:
    """Find the shortest path to any of the given goals.

    Args:
        game_state: Current game state.
        start: Starting position.
        goals: List of goal positions.
        danger_map: Pre-computed danger map.
        **kwargs: Additional arguments passed to find_path.

    Returns:
        Tuple of (path, reached_goal) or (None, None) if no path found.
    """
    if danger_map is None:
        danger_map = DangerMap(game_state)

    best_path = None
    best_goal = None
    best_cost = float('inf')

    for goal in goals:
        path = find_path(game_state, start, goal, danger_map, **kwargs)
        if path and len(path) < best_cost:
            best_path = path
            best_goal = goal
            best_cost = len(path)

    return best_path, best_goal


def find_safe_tiles(
    game_state: GameState,
    start: Point,
    danger_map: Optional[DangerMap] = None,
    max_distance: int = 10
) -> List[Point]:
    """Find all safe tiles reachable from start within max_distance.

    Uses BFS to explore reachable safe tiles.

    Args:
        game_state: Current game state.
        start: Starting position.
        danger_map: Pre-computed danger map.
        max_distance: Maximum search distance.

    Returns:
        List of safe reachable Points.
    """
    if danger_map is None:
        danger_map = DangerMap(game_state)

    from collections import deque

    width = game_state.world.width
    height = game_state.world.height

    safe_tiles: List[Point] = []
    queue = deque([(start.x, start.y, 0)])
    visited = {(start.x, start.y)}

    while queue:
        cx, cy, dist = queue.popleft()

        # Check if current tile is safe (not including start if in danger)
        if not danger_map.is_dangerous(cx, cy) and (cx, cy) != (start.x, start.y):
            safe_tiles.append(Point(cx, cy))

        if dist >= max_distance:
            continue

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < width and 0 <= ny < height):
                continue

            if (nx, ny) in visited:
                continue

            # Check walkability (ignore bombs for escape planning)
            if not game_state.is_walkable(nx, ny, ignore_units=True, ignore_bombs=True):
                continue

            visited.add((nx, ny))
            queue.append((nx, ny, dist + 1))

    return safe_tiles


def find_nearest_safe_tile(
    game_state: GameState,
    start: Point,
    danger_map: Optional[DangerMap] = None
) -> Optional[Point]:
    """Find the nearest safe tile from start.

    Args:
        game_state: Current game state.
        start: Starting position.
        danger_map: Pre-computed danger map.

    Returns:
        Nearest safe Point, or None if none found.
    """
    safe_tiles = find_safe_tiles(game_state, start, danger_map, max_distance=15)
    if not safe_tiles:
        return None

    # Sort by distance
    safe_tiles.sort(key=lambda p: abs(p.x - start.x) + abs(p.y - start.y))
    return safe_tiles[0]


def calculate_path_danger(
    path: List[Point],
    danger_map: DangerMap,
    start_tick: int
) -> int:
    """Calculate the total danger exposure along a path.

    Considers that each step takes one tick.

    Args:
        path: Path to evaluate.
        danger_map: Pre-computed danger map.
        start_tick: Starting tick.

    Returns:
        Total danger score for the path.
    """
    total_danger = 0

    for i, point in enumerate(path):
        tick = start_tick + i
        if not danger_map.is_safe_at_tick(point.x, point.y, tick):
            danger_level = danger_map.get_danger_level(point.x, point.y)
            total_danger += danger_level

    return total_danger


def find_escape_path(
    game_state: GameState,
    unit: UnitState,
    danger_map: Optional[DangerMap] = None
) -> Optional[List[Point]]:
    """Find the best escape path for a unit in danger.

    Prioritizes paths that reach safety fastest while minimizing danger exposure.

    Args:
        game_state: Current game state.
        unit: Unit that needs to escape.
        danger_map: Pre-computed danger map.

    Returns:
        Escape path, or None if no escape found.
    """
    if danger_map is None:
        danger_map = DangerMap(game_state)

    start = Point(unit.x, unit.y)

    # If not in danger, no need to escape
    if not danger_map.is_dangerous(unit.x, unit.y):
        return None

    # Find nearest safe tile
    nearest_safe = find_nearest_safe_tile(game_state, start, danger_map)
    if nearest_safe is None:
        return None

    # Find path to it (allow some danger to escape)
    path = find_path(
        game_state,
        start,
        nearest_safe,
        danger_map,
        avoid_danger=False,  # We need to pass through danger to escape
        allow_destruction=False,  # No time to destroy blocks
        avoid_units=False  # Desperate times
    )

    return path
