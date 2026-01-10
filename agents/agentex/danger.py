"""
Danger zone calculation for the AgentEx Bomberland agent.

This module handles:
- Computing tiles threatened by bombs (current and future)
- Chain detonation analysis
- Safety evaluation for movement planning
- Blast timeline prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from types_ import Entity, EntityType, GameState, Point, UnitState

# Engine constants
BOMB_ARMED_TICKS = 5
BOMB_DURATION_TICKS = 30
BLAST_DURATION_TICKS = 5


@dataclass
class DangerInfo:
    """Information about danger at a specific tile.

    Attributes:
        is_dangerous: Whether the tile is currently dangerous.
        danger_start_tick: Earliest tick when danger could occur (armed bomb).
        danger_end_tick: Latest tick when danger clears (blast fades).
        source_bombs: Bombs that threaten this tile.
        is_active_blast: Whether there's currently an active blast.
        is_fire: Whether this tile has endgame fire.
    """

    is_dangerous: bool = False
    danger_start_tick: Optional[int] = None
    danger_end_tick: Optional[int] = None
    source_bombs: List[Point] = field(default_factory=list)
    is_active_blast: bool = False
    is_fire: bool = False


class DangerMap:
    """Manages danger zone calculations for the game map.

    Caches danger information for efficient lookups during pathfinding
    and decision making.
    """

    def __init__(self, game_state: GameState):
        """Initialize danger map from current game state.

        Args:
            game_state: Current game state to analyze.
        """
        self.game_state = game_state
        self.tick = game_state.tick
        self.width = game_state.world.width
        self.height = game_state.world.height

        # Cache of danger info per tile
        self._danger_cache: Dict[Tuple[int, int], DangerInfo] = {}

        # Precompute danger zones
        self._compute_danger_zones()

    def _compute_danger_zones(self) -> None:
        """Precompute danger zones for all bombs on the map."""
        bombs = [e for e in self.game_state.entities if e.entity_type == EntityType.BOMB]
        blasts = [e for e in self.game_state.entities if e.entity_type == EntityType.BLAST]

        # Mark active blasts
        for blast in blasts:
            key = (blast.x, blast.y)
            if key not in self._danger_cache:
                self._danger_cache[key] = DangerInfo()
            info = self._danger_cache[key]
            info.is_dangerous = True
            info.is_active_blast = True
            if blast.expires:
                info.danger_end_tick = blast.expires

            # Check if it's endgame fire (no owner)
            if blast.owner_unit_id is None:
                info.is_fire = True
                info.danger_end_tick = None  # Fire is permanent

        # Process each bomb and compute blast zones
        for bomb in bombs:
            self._add_bomb_danger_zone(bomb, set())

    def _add_bomb_danger_zone(
        self,
        bomb: Entity,
        visited: Set[Tuple[int, int]],
        triggered_tick: Optional[int] = None
    ) -> None:
        """Add danger zones for a bomb, including chain reactions.

        Args:
            bomb: The bomb entity to process.
            visited: Set of bomb positions already processed (for cycle detection).
            triggered_tick: If this bomb was triggered by another, when that happens.
        """
        bomb_pos = (bomb.x, bomb.y)
        if bomb_pos in visited:
            return
        visited.add(bomb_pos)

        # Calculate when this bomb becomes dangerous
        armed_tick = bomb.created + BOMB_ARMED_TICKS
        auto_explode_tick = bomb.expires if bomb.expires else bomb.created + BOMB_DURATION_TICKS

        # If triggered by chain, use trigger time if it's before auto-explode
        if triggered_tick and triggered_tick >= armed_tick:
            effective_explode_tick = min(triggered_tick, auto_explode_tick)
        else:
            effective_explode_tick = auto_explode_tick

        # Calculate blast end time
        blast_end_tick = effective_explode_tick + BLAST_DURATION_TICKS

        # Get blast radius from bomb or owner unit
        blast_radius = self._get_bomb_blast_radius(bomb)

        # Mark all tiles in blast zone
        blast_tiles = self._get_blast_tiles(bomb.x, bomb.y, blast_radius)

        for tile_x, tile_y in blast_tiles:
            key = (tile_x, tile_y)
            if key not in self._danger_cache:
                self._danger_cache[key] = DangerInfo()

            info = self._danger_cache[key]
            info.is_dangerous = True
            info.source_bombs.append(Point(bomb.x, bomb.y))

            # Update danger timing (worst case)
            if info.danger_start_tick is None or armed_tick < info.danger_start_tick:
                info.danger_start_tick = armed_tick
            if info.danger_end_tick is None or blast_end_tick > info.danger_end_tick:
                info.danger_end_tick = blast_end_tick

        # Check for chain reactions - other bombs in blast zone
        for tile_x, tile_y in blast_tiles:
            other_bombs = [
                e for e in self.game_state.entities
                if e.entity_type == EntityType.BOMB
                and e.x == tile_x and e.y == tile_y
                and (e.x, e.y) != bomb_pos
            ]
            for other_bomb in other_bombs:
                # Chain reaction happens when this bomb explodes
                self._add_bomb_danger_zone(other_bomb, visited, effective_explode_tick)

    def _get_bomb_blast_radius(self, bomb: Entity) -> int:
        """Get the blast radius for a bomb.

        Args:
            bomb: The bomb entity.

        Returns:
            Blast radius in tiles.
        """
        diameter = bomb.blast_diameter
        if diameter is None:
            # Try to get from owner unit
            if bomb.owner_unit_id:
                owner = self.game_state.get_unit(bomb.owner_unit_id)
                if owner:
                    diameter = owner.blast_diameter
        if diameter is None:
            diameter = 3  # Default

        return max(0, (diameter - 1) // 2)

    def _get_blast_tiles(self, x: int, y: int, radius: int) -> List[Tuple[int, int]]:
        """Get all tiles affected by a blast centered at (x, y).

        Blast extends in cardinal directions but stops at solid blocks.

        Args:
            x: Bomb x coordinate.
            y: Bomb y coordinate.
            radius: Blast radius in tiles.

        Returns:
            List of (x, y) tuples in the blast zone.
        """
        tiles = [(x, y)]  # Center is always affected

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for dist in range(1, radius + 1):
                nx, ny = x + dx * dist, y + dy * dist

                # Check bounds
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    break

                tiles.append((nx, ny))

                # Check for blocking entities (blast stops after hitting them)
                entities_here = self.game_state.entities_at(nx, ny)
                if any(e.entity_type in {
                    EntityType.METAL_BLOCK,
                    EntityType.ORE_BLOCK,
                    EntityType.WOOD_BLOCK
                } for e in entities_here):
                    break

        return tiles

    def get_danger_info(self, x: int, y: int) -> DangerInfo:
        """Get danger information for a specific tile.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            DangerInfo for the tile.
        """
        return self._danger_cache.get((x, y), DangerInfo())

    def is_dangerous(self, x: int, y: int) -> bool:
        """Check if a tile is in any danger zone.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if tile is dangerous, False otherwise.
        """
        return self.get_danger_info(x, y).is_dangerous

    def is_immediately_dangerous(self, x: int, y: int) -> bool:
        """Check if a tile has immediate danger (active blast or fire).

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if tile has immediate danger.
        """
        info = self.get_danger_info(x, y)
        return info.is_active_blast or info.is_fire

    def is_safe_at_tick(self, x: int, y: int, tick: int) -> bool:
        """Check if a tile will be safe at a specific future tick.

        Args:
            x: X coordinate.
            y: Y coordinate.
            tick: Future tick to check.

        Returns:
            True if tile will be safe at that tick.
        """
        info = self.get_danger_info(x, y)

        # Never safe if fire
        if info.is_fire:
            return False

        # Not dangerous at all
        if not info.is_dangerous:
            return True

        # Check if we're before danger starts or after it ends
        if info.danger_start_tick and tick < info.danger_start_tick:
            return True
        if info.danger_end_tick and tick >= info.danger_end_tick:
            return True

        return False

    def get_danger_level(self, x: int, y: int) -> int:
        """Get a numeric danger level for a tile.

        Higher values indicate more danger. Useful for pathfinding costs.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Danger level: 0=safe, 50=bomb zone (not armed), 100=armed bomb, 1000=active blast/fire
        """
        info = self.get_danger_info(x, y)

        if info.is_fire:
            return 1000

        if info.is_active_blast:
            return 1000

        if not info.is_dangerous:
            return 0

        # Check if any source bomb is armed
        current_tick = self.tick
        if info.danger_start_tick and current_tick >= info.danger_start_tick:
            return 100  # Armed bomb zone

        return 50  # Bomb zone but not yet armed

    def get_ticks_until_safe(self, x: int, y: int) -> Optional[int]:
        """Get number of ticks until a tile becomes safe.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Ticks until safe, None if never safe (fire), 0 if already safe.
        """
        info = self.get_danger_info(x, y)

        if info.is_fire:
            return None

        if not info.is_dangerous:
            return 0

        if info.danger_end_tick:
            ticks = info.danger_end_tick - self.tick
            return max(0, ticks)

        return None

    def find_safe_neighbors(
        self,
        x: int,
        y: int,
        must_be_walkable: bool = True
    ) -> List[Tuple[int, int]]:
        """Find safe neighboring tiles.

        Args:
            x: X coordinate.
            y: Y coordinate.
            must_be_walkable: If True, also check that tiles are walkable.

        Returns:
            List of safe neighboring (x, y) coordinates.
        """
        safe_neighbors = []

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue

            if self.is_dangerous(nx, ny):
                continue

            if must_be_walkable:
                if not self.game_state.is_walkable(nx, ny, ignore_units=True):
                    continue

            safe_neighbors.append((nx, ny))

        return safe_neighbors


def calculate_escape_routes(
    game_state: GameState,
    unit: UnitState,
    danger_map: Optional[DangerMap] = None
) -> List[List[Point]]:
    """Calculate possible escape routes for a unit in danger.

    Uses BFS to find paths to safe tiles outside all danger zones.

    Args:
        game_state: Current game state.
        unit: Unit that needs to escape.
        danger_map: Pre-computed danger map (will create if not provided).

    Returns:
        List of escape paths (each path is a list of Points from current position).
    """
    if danger_map is None:
        danger_map = DangerMap(game_state)

    start = (unit.x, unit.y)

    # If already safe, no need to escape
    if not danger_map.is_dangerous(unit.x, unit.y):
        return [[Point(unit.x, unit.y)]]

    # BFS to find escape routes
    from collections import deque

    queue = deque([(start, [Point(unit.x, unit.y)])])
    visited = {start}
    escape_routes = []

    while queue:
        (cx, cy), path = queue.popleft()

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < game_state.world.width and 0 <= ny < game_state.world.height):
                continue

            if (nx, ny) in visited:
                continue

            # Check if walkable - do NOT ignore bombs (can't walk onto bomb tiles)
            if not game_state.is_walkable(nx, ny, ignore_units=True, ignore_bombs=False):
                continue

            visited.add((nx, ny))
            new_path = path + [Point(nx, ny)]

            # Check if this is a safe tile
            if not danger_map.is_dangerous(nx, ny):
                escape_routes.append(new_path)
                # Don't stop - collect multiple routes
            else:
                # Continue searching
                queue.append(((nx, ny), new_path))

    # Sort by path length (shortest first)
    escape_routes.sort(key=len)

    return escape_routes


def can_escape_after_bomb(
    game_state: GameState,
    unit: UnitState,
    bomb_position: Optional[Point] = None,
    max_escape_moves: int = 3
) -> Tuple[bool, Optional[List[Point]]]:
    """Check if a unit can escape after placing a bomb.

    Args:
        game_state: Current game state.
        unit: Unit that would place the bomb.
        bomb_position: Where the bomb would be placed (default: unit position).
        max_escape_moves: Maximum number of moves allowed to escape (default 3).
            This ensures the unit can reach safety quickly rather than finding
            any path that eventually exits the blast zone.

    Returns:
        Tuple of (can_escape, escape_path).
    """
    if bomb_position is None:
        bomb_position = Point(unit.x, unit.y)

    # Simulate the bomb placement
    blast_radius = max(0, (unit.blast_diameter - 1) // 2)

    # Get tiles in blast zone
    blast_tiles: Set[Tuple[int, int]] = {(bomb_position.x, bomb_position.y)}

    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for dist in range(1, blast_radius + 1):
            nx, ny = bomb_position.x + dx * dist, bomb_position.y + dy * dist

            if not (0 <= nx < game_state.world.width and 0 <= ny < game_state.world.height):
                break

            blast_tiles.add((nx, ny))

            # Check for blocking entities
            entities_here = game_state.entities_at(nx, ny)
            if any(e.entity_type in {
                EntityType.METAL_BLOCK,
                EntityType.ORE_BLOCK,
                EntityType.WOOD_BLOCK
            } for e in entities_here):
                break

    # BFS to find escape route (with move limit)
    from collections import deque

    start = (unit.x, unit.y)
    queue = deque([(start, [Point(unit.x, unit.y)])])
    visited = {start}

    while queue:
        (cx, cy), path = queue.popleft()

        # Check if we've exceeded max moves (path includes starting position)
        num_moves = len(path) - 1
        if num_moves >= max_escape_moves:
            continue  # Don't explore further from this path

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < game_state.world.width and 0 <= ny < game_state.world.height):
                continue

            if (nx, ny) in visited:
                continue

            # Skip the bomb position itself
            if nx == bomb_position.x and ny == bomb_position.y:
                continue

            # Check if walkable - do NOT ignore bombs (can't walk onto bomb tiles)
            # but ignore units (they might move)
            if not game_state.is_walkable(nx, ny, ignore_units=True, ignore_bombs=False):
                continue

            visited.add((nx, ny))
            new_path = path + [Point(nx, ny)]

            # Check if outside blast zone
            if (nx, ny) not in blast_tiles:
                return True, new_path

            queue.append(((nx, ny), new_path))

    return False, None


def unit_in_danger(game_state: GameState, unit: UnitState, danger_map: Optional[DangerMap] = None) -> bool:
    """Check if a unit is currently in a danger zone.

    Args:
        game_state: Current game state.
        unit: Unit to check.
        danger_map: Pre-computed danger map (will create if not provided).

    Returns:
        True if unit is in danger.
    """
    if danger_map is None:
        danger_map = DangerMap(game_state)

    return danger_map.is_dangerous(unit.x, unit.y)
