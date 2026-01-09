"""
Endgame fire handling for the AgentEx Bomberland agent.

This module handles:
- Tracking the ring of fire progression
- Predicting fire spawn locations and timing
- Positioning strategy during endgame
- Survival prioritization when fire closes in
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from types_ import GameState, Point

# Engine constants for endgame
ENDGAME_START_TICK = 200
FIRE_SPAWN_INTERVAL = 2  # Fire spawns every 2 ticks


@dataclass
class FirePrediction:
    """Prediction for when fire will reach a tile.

    Attributes:
        position: The tile position.
        arrival_tick: Tick when fire will spawn here.
        is_on_fire: Whether tile currently has fire.
    """

    position: Point
    arrival_tick: Optional[int]
    is_on_fire: bool


class FireTracker:
    """Tracks and predicts the endgame fire spiral.

    The fire starts at corners (0,0) and (14,14) on tick 200,
    then spirals inward, spawning every 2 ticks.
    """

    def __init__(self, game_state: GameState):
        """Initialize fire tracker.

        Args:
            game_state: Current game state.
        """
        self.game_state = game_state
        self.width = game_state.world.width
        self.height = game_state.world.height
        self.tick = game_state.tick

        # Pre-compute fire spiral pattern
        self._fire_pattern: List[Tuple[int, int]] = []
        self._fire_timing: Dict[Tuple[int, int], int] = {}
        self._compute_fire_pattern()

        # Track current fire tiles
        self._current_fire_tiles: Set[Tuple[int, int]] = set()
        self._update_current_fire()

    def _compute_fire_pattern(self) -> None:
        """Compute the fire spiral pattern and timing.

        Fire spawns from corners toward center in a spiral pattern.
        The exact pattern starts from top-left and bottom-right,
        moving horizontally first, then vertically.
        """
        # Generate spiral pattern from both corners simultaneously
        # This is an approximation of the actual engine behavior

        visited: Set[Tuple[int, int]] = set()
        pattern: List[Tuple[int, int]] = []

        # We'll simulate two spirals converging
        # Top-left corner spiral
        tl_x, tl_y = 0, self.height - 1  # Top-left in game coords
        # Bottom-right corner spiral
        br_x, br_y = self.width - 1, 0  # Bottom-right in game coords

        # Simple spiral approximation
        # Move inward layer by layer
        for layer in range(max(self.width, self.height) // 2 + 1):
            # Top edge (left to right) for layer
            for x in range(layer, self.width - layer):
                y = self.height - 1 - layer
                if (x, y) not in visited and 0 <= y < self.height:
                    visited.add((x, y))
                    pattern.append((x, y))

            # Bottom edge (right to left) for layer
            for x in range(self.width - 1 - layer, layer - 1, -1):
                y = layer
                if (x, y) not in visited and 0 <= y < self.height:
                    visited.add((x, y))
                    pattern.append((x, y))

            # Right edge (top to bottom) for layer
            for y in range(self.height - 2 - layer, layer, -1):
                x = self.width - 1 - layer
                if (x, y) not in visited and 0 <= x < self.width:
                    visited.add((x, y))
                    pattern.append((x, y))

            # Left edge (bottom to top) for layer
            for y in range(layer + 1, self.height - 1 - layer):
                x = layer
                if (x, y) not in visited and 0 <= x < self.width:
                    visited.add((x, y))
                    pattern.append((x, y))

        self._fire_pattern = pattern

        # Calculate timing for each tile
        for i, pos in enumerate(self._fire_pattern):
            tick = ENDGAME_START_TICK + (i * FIRE_SPAWN_INTERVAL)
            self._fire_timing[pos] = tick

    def _update_current_fire(self) -> None:
        """Update set of tiles currently on fire."""
        from types_ import EntityType

        self._current_fire_tiles.clear()
        for entity in self.game_state.entities:
            # Fire appears as blast entities without owner
            if entity.entity_type == EntityType.BLAST and entity.owner_unit_id is None:
                self._current_fire_tiles.add((entity.x, entity.y))

    def is_on_fire(self, x: int, y: int) -> bool:
        """Check if a tile is currently on fire.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if tile has fire.
        """
        return (x, y) in self._current_fire_tiles

    def get_fire_arrival_tick(self, x: int, y: int) -> Optional[int]:
        """Get the tick when fire will arrive at a tile.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Arrival tick, or None if tile won't get fire (shouldn't happen).
        """
        return self._fire_timing.get((x, y))

    def is_safe_from_fire(self, x: int, y: int, at_tick: Optional[int] = None) -> bool:
        """Check if a tile is safe from fire at a given tick.

        Args:
            x: X coordinate.
            y: Y coordinate.
            at_tick: Tick to check (default: current tick).

        Returns:
            True if tile is safe from fire.
        """
        if at_tick is None:
            at_tick = self.tick

        # Already on fire
        if self.is_on_fire(x, y):
            return False

        # Check predicted arrival
        arrival = self.get_fire_arrival_tick(x, y)
        if arrival is None:
            return True  # No fire predicted

        return at_tick < arrival

    def get_ticks_until_fire(self, x: int, y: int) -> Optional[int]:
        """Get number of ticks until fire reaches a tile.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Ticks until fire, None if already on fire or no fire predicted.
        """
        if self.is_on_fire(x, y):
            return 0

        arrival = self.get_fire_arrival_tick(x, y)
        if arrival is None:
            return None

        ticks = arrival - self.tick
        return max(0, ticks)

    def get_safe_tiles(self, at_tick: Optional[int] = None) -> List[Tuple[int, int]]:
        """Get all tiles that are safe from fire at given tick.

        Args:
            at_tick: Tick to check (default: current tick).

        Returns:
            List of (x, y) tuples for safe tiles.
        """
        if at_tick is None:
            at_tick = self.tick

        safe_tiles = []
        for x in range(self.width):
            for y in range(self.height):
                if self.is_safe_from_fire(x, y, at_tick):
                    safe_tiles.append((x, y))

        return safe_tiles

    def get_center_distance(self, x: int, y: int) -> int:
        """Get Manhattan distance from a tile to map center.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Distance to center.
        """
        center_x = self.width // 2
        center_y = self.height // 2
        return abs(x - center_x) + abs(y - center_y)

    def get_prediction(self, x: int, y: int) -> FirePrediction:
        """Get full fire prediction for a tile.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            FirePrediction for the tile.
        """
        return FirePrediction(
            position=Point(x, y),
            arrival_tick=self.get_fire_arrival_tick(x, y),
            is_on_fire=self.is_on_fire(x, y)
        )


def get_endgame_target_position(
    game_state: GameState,
    unit_position: Point,
    fire_tracker: Optional[FireTracker] = None
) -> Optional[Point]:
    """Get the best position to move toward during endgame.

    Prioritizes:
    1. Staying away from fire
    2. Moving toward center
    3. Avoiding dangerous tiles

    Args:
        game_state: Current game state.
        unit_position: Current unit position.
        fire_tracker: Pre-computed fire tracker.

    Returns:
        Target position, or None if no good option.
    """
    if fire_tracker is None:
        fire_tracker = FireTracker(game_state)

    width = game_state.world.width
    height = game_state.world.height
    center = Point(width // 2, height // 2)

    # Get safe tiles
    safe_tiles = fire_tracker.get_safe_tiles()

    if not safe_tiles:
        return None  # No safe tiles - game is ending

    # Filter walkable tiles
    walkable_safe = [
        (x, y) for x, y in safe_tiles
        if game_state.is_walkable(x, y, ignore_units=True, ignore_bombs=True)
    ]

    if not walkable_safe:
        # Fall back to any safe tile
        walkable_safe = safe_tiles

    # Score tiles by center distance (lower is better)
    def tile_score(pos: Tuple[int, int]) -> Tuple[int, int]:
        x, y = pos
        center_dist = abs(x - center.x) + abs(y - center.y)
        # Also consider time until fire
        fire_time = fire_tracker.get_ticks_until_fire(x, y)
        if fire_time is None:
            fire_time = 1000  # No fire predicted

        # Prefer tiles closer to center with more time before fire
        return (-fire_time, center_dist)  # Negative because we want max fire_time

    walkable_safe.sort(key=tile_score)

    if walkable_safe:
        best = walkable_safe[0]
        return Point(best[0], best[1])

    return center  # Fall back to center


def should_prioritize_survival(game_state: GameState) -> bool:
    """Check if survival should be prioritized over attacking.

    During endgame, survival becomes more important as safe space shrinks.

    Args:
        game_state: Current game state.

    Returns:
        True if survival should be prioritized.
    """
    if game_state.tick < ENDGAME_START_TICK:
        return False

    # In endgame, prioritize survival
    fire_tracker = FireTracker(game_state)

    # Count safe tiles
    safe_tiles = fire_tracker.get_safe_tiles()

    # If less than 50 safe tiles, survival is critical
    if len(safe_tiles) < 50:
        return True

    # If fire is within 5 ticks of any of our units
    for unit in game_state.my_alive_units:
        fire_time = fire_tracker.get_ticks_until_fire(unit.x, unit.y)
        if fire_time is not None and fire_time <= 5:
            return True

    return False


def get_fire_safe_path(
    game_state: GameState,
    start: Point,
    goal: Point,
    fire_tracker: Optional[FireTracker] = None
) -> Optional[List[Point]]:
    """Find a path that avoids fire (current and predicted).

    Args:
        game_state: Current game state.
        start: Starting position.
        goal: Goal position.
        fire_tracker: Pre-computed fire tracker.

    Returns:
        Path avoiding fire, or None if no safe path.
    """
    from pathfinding import find_path
    from danger import DangerMap

    if fire_tracker is None:
        fire_tracker = FireTracker(game_state)

    danger_map = DangerMap(game_state)

    # Create excluded positions set for tiles that will have fire soon
    excluded: Set[Tuple[int, int]] = set()

    current_tick = game_state.tick

    # Estimate path length to know which tiles to exclude
    estimated_length = abs(goal.x - start.x) + abs(goal.y - start.y)

    for x in range(game_state.world.width):
        for y in range(game_state.world.height):
            fire_time = fire_tracker.get_ticks_until_fire(x, y)
            if fire_time is not None and fire_time <= estimated_length:
                excluded.add((x, y))

    return find_path(
        game_state,
        start,
        goal,
        danger_map,
        excluded_positions=excluded,
        avoid_danger=True,
        allow_destruction=False
    )
