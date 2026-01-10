"""
AgentEx - Tournament-Competitive Rule-Based Bomberland Agent

This is the main agent implementation that coordinates all strategic components
to make decisions each game tick.

Key features:
- Safety-first decision making with danger zone awareness
- Three-phase strategy (early/mid/endgame)
- Intelligent bomb placement with retreat planning
- Powerup collection prioritization
- Unit coordination to avoid conflicts
- Endgame fire handling and center positioning
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Dict, Optional, Set

from game_state import GameState as GameStateClient
from types_ import (
    ActionPacket,
    BombAction,
    DetonateAction,
    GameState,
    MoveAction,
    Point,
    SkipAction,
    UnitState,
)
from danger import DangerMap, unit_in_danger
from pathfinding import find_path, find_escape_path, find_nearest_safe_tile
from strategy import (
    ActionPriority,
    GamePhase,
    StrategyManager,
    UnitDecision,
    get_game_phase,
)
from bomb_logic import check_immediate_detonation, evaluate_bomb_placement
from endgame import FireTracker, get_endgame_target_position, should_prioritize_survival
from utils import can_place_bomb, get_direction_to_point, manhattan_distance, is_position_blocked

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[agentex] %(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Connection string from environment
URI = (
    os.environ.get("GAME_CONNECTION_STRING")
    or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=agentex"
)


class Agent:
    """Main agent class coordinating all strategic components."""

    def __init__(self):
        """Initialize the agent and connect to game server."""
        self._client = GameStateClient(URI)
        self._client.set_game_tick_callback(self._on_game_tick)

        # Track last positions to detect stuck units
        self._last_positions: Dict[str, Point] = {}
        self._stuck_counters: Dict[str, int] = {}

        # Reserved positions for coordination
        self._reserved_positions: Set[tuple] = set()

        # Start connection
        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [
            asyncio.ensure_future(self._client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))

    async def _on_game_tick(self, tick_number: int, raw_game_state: dict) -> None:
        """Handle each game tick.

        Args:
            tick_number: Current game tick.
            raw_game_state: Raw game state from server.
        """
        # Parse game state
        game_state = GameState.from_dict(raw_game_state)

        # Log tick info periodically
        if tick_number % 50 == 0:
            logger.info(
                f"Tick {tick_number} | Phase: {get_game_phase(tick_number).name} | "
                f"My units: {len(game_state.my_alive_units)} | "
                f"Enemy units: {len(game_state.enemy_alive_units)}"
            )

        # Skip if no units alive
        if not game_state.my_alive_units:
            return

        # Clear stale data
        self._reserved_positions.clear()

        # Create danger map for this tick
        danger_map = DangerMap(game_state)

        # Get game phase
        phase = get_game_phase(tick_number)

        # Compute strategy decisions for ALL units at once (not per-unit)
        strategy = StrategyManager(game_state)
        decisions = strategy.compute_decisions()

        # Process each unit
        for unit in game_state.my_alive_units:
            try:
                action = await self._decide_unit_action(
                    game_state, unit, danger_map, phase, tick_number, decisions
                )
                if action:
                    await self._send_action(action)
            except Exception as e:
                logger.error(f"Error processing unit {unit.unit_id}: {e}")

    async def _decide_unit_action(
        self,
        game_state: GameState,
        unit: UnitState,
        danger_map: DangerMap,
        phase: GamePhase,
        tick: int,
        decisions: Dict[str, 'UnitDecision']
    ) -> Optional[ActionPacket]:
        """Decide what action a unit should take.

        Args:
            game_state: Current game state.
            unit: Unit to decide for.
            danger_map: Pre-computed danger map.
            phase: Current game phase.
            tick: Current tick.
            decisions: Pre-computed decisions from StrategyManager.

        Returns:
            Action to execute, or None.
        """
        unit_id = unit.unit_id

        # Priority 1: ESCAPE if in danger
        if unit_in_danger(game_state, unit, danger_map):
            escape_action = self._handle_escape(game_state, unit, danger_map)
            if escape_action:
                logger.info(f"Unit {unit_id} escaping danger")
                return escape_action

        # Priority 2: DETONATE if enemy in blast zone (and we're safe)
        detonate_pos = check_immediate_detonation(game_state, unit)
        if detonate_pos:
            logger.info(f"Unit {unit_id} detonating bomb at {detonate_pos} to hit enemy")
            return DetonateAction(unit_id=unit_id, target=detonate_pos)

        # Priority 3: Phase-specific strategy
        if phase == GamePhase.ENDGAME:
            action = self._handle_endgame(game_state, unit, danger_map)
            if action:
                return action

        # Use pre-computed decisions
        if unit_id in decisions:
            decision = decisions[unit_id]
            # Log decision details for debugging
            if decision.target:
                target_info = f"target={decision.target.target_type}@{decision.target.position}"
                path_info = f"path_len={len(decision.target.path) if decision.target.path else 0}"
            else:
                target_info = "no_target"
                path_info = ""
            logger.debug(f"Unit {unit_id} decision: {decision.action_type} {target_info} {path_info}")
            return self._execute_decision(game_state, unit, decision, danger_map, phase)
        else:
            logger.debug(f"Unit {unit_id} has no decision from strategy manager")

        # Fallback: Random safe movement
        return self._handle_fallback(game_state, unit, danger_map)

    def _handle_escape(
        self,
        game_state: GameState,
        unit: UnitState,
        danger_map: DangerMap
    ) -> Optional[ActionPacket]:
        """Handle escape when unit is in danger.

        Args:
            game_state: Current game state.
            unit: Unit in danger.
            danger_map: Pre-computed danger map.

        Returns:
            Escape action, or None.
        """
        escape_path = find_escape_path(game_state, unit, danger_map)

        if escape_path and len(escape_path) > 1:
            next_pos = escape_path[1]
            direction = get_direction_to_point(Point(unit.x, unit.y), next_pos)
            if direction:
                return MoveAction(unit_id=unit.unit_id, move=direction)

        # No escape path - check if we can detonate something
        detonate_pos = check_immediate_detonation(game_state, unit)
        if detonate_pos:
            return DetonateAction(unit_id=unit.unit_id, target=detonate_pos)

        return None

    def _handle_endgame(
        self,
        game_state: GameState,
        unit: UnitState,
        danger_map: DangerMap
    ) -> Optional[ActionPacket]:
        """Handle endgame strategy.

        Args:
            game_state: Current game state.
            unit: Unit to control.
            danger_map: Pre-computed danger map.

        Returns:
            Action for endgame, or None to fall through.
        """
        fire_tracker = FireTracker(game_state)

        # Check if we need to prioritize survival
        if should_prioritize_survival(game_state):
            # Get target position toward center/away from fire
            target = get_endgame_target_position(
                game_state,
                Point(unit.x, unit.y),
                fire_tracker
            )

            if target and (target.x != unit.x or target.y != unit.y):
                path = find_path(
                    game_state,
                    Point(unit.x, unit.y),
                    target,
                    danger_map,
                    allow_destruction=False
                )

                if path and len(path) > 1:
                    next_pos = path[1]
                    direction = get_direction_to_point(Point(unit.x, unit.y), next_pos)
                    if direction:
                        logger.info(f"Unit {unit.unit_id} moving toward safe zone")
                        return MoveAction(unit_id=unit.unit_id, move=direction)

        return None

    def _execute_decision(
        self,
        game_state: GameState,
        unit: UnitState,
        decision: UnitDecision,
        danger_map: DangerMap,
        phase: GamePhase
    ) -> Optional[ActionPacket]:
        """Execute a strategic decision.

        Args:
            game_state: Current game state.
            unit: Unit to control.
            decision: Decision from strategy manager.
            danger_map: Pre-computed danger map.
            phase: Current game phase.

        Returns:
            Action to execute.
        """
        unit_id = unit.unit_id

        if decision.action_type == 'escape' and decision.next_position:
            # Validate move is possible
            if self._is_move_valid(game_state, unit, decision.next_position):
                direction = get_direction_to_point(Point(unit.x, unit.y), decision.next_position)
                if direction:
                    return MoveAction(unit_id=unit_id, move=direction)

        elif decision.action_type == 'move' and decision.next_position:
            # Reserve position
            pos_key = (decision.next_position.x, decision.next_position.y)
            if pos_key in self._reserved_positions:
                return None  # Position taken

            # Validate move is possible
            if not self._is_move_valid(game_state, unit, decision.next_position):
                return None

            self._reserved_positions.add(pos_key)
            direction = get_direction_to_point(Point(unit.x, unit.y), decision.next_position)
            if direction:
                return MoveAction(unit_id=unit_id, move=direction)

        elif decision.action_type == 'bomb' and decision.place_bomb:
            # Evaluate bomb placement - only place if safe escape exists
            bomb_plan = evaluate_bomb_placement(game_state, unit, danger_map)
            if bomb_plan and can_place_bomb(game_state, unit):
                logger.info(f"Unit {unit_id} placing bomb")
                return BombAction(unit_id=unit_id)

        elif decision.action_type == 'detonate' and decision.detonate_position:
            return DetonateAction(unit_id=unit_id, target=decision.detonate_position)

        return None

    def _is_move_valid(
        self,
        game_state: GameState,
        unit: UnitState,
        next_pos: Point
    ) -> bool:
        """Check if a move to next_pos is actually valid (tile is walkable).

        Args:
            game_state: Current game state.
            unit: Unit trying to move.
            next_pos: Target position.

        Returns:
            True if move is valid, False otherwise.
        """
        # Check if position is adjacent
        dx = abs(next_pos.x - unit.x)
        dy = abs(next_pos.y - unit.y)
        if dx + dy != 1:
            return False

        # Check if position is blocked
        if is_position_blocked(game_state, next_pos.x, next_pos.y, ignore_units=False, ignore_bombs=False):
            return False

        # Check if reserved by another unit
        if (next_pos.x, next_pos.y) in self._reserved_positions:
            return False

        return True

    def _handle_fallback(
        self,
        game_state: GameState,
        unit: UnitState,
        danger_map: DangerMap
    ) -> Optional[ActionPacket]:
        """Handle fallback when no clear strategy.

        Args:
            game_state: Current game state.
            unit: Unit to control.
            danger_map: Pre-computed danger map.

        Returns:
            Fallback action.
        """
        from strategy import (
            get_blocking_entity_at,
            should_place_bomb_to_clear,
            has_bomb_targeting_position
        )

        unit_id = unit.unit_id
        current = Point(unit.x, unit.y)

        # Try to find nearest enemy and move toward them
        if game_state.enemy_alive_units:
            nearest_enemy = min(
                game_state.enemy_alive_units,
                key=lambda e: manhattan_distance(current, e.position)
            )

            # First try without destruction (direct walkable path)
            path = find_path(
                game_state,
                current,
                nearest_enemy.position,
                danger_map,
                allow_destruction=False  # Only walkable tiles
            )

            if path and len(path) > 1:
                next_pos = path[1]

                # Validate the move is actually possible
                if self._is_move_valid(game_state, unit, next_pos):
                    self._reserved_positions.add((next_pos.x, next_pos.y))
                    direction = get_direction_to_point(current, next_pos)
                    if direction:
                        return MoveAction(unit_id=unit_id, move=direction)

            # No direct path - try with destruction
            path_with_destruction = find_path(
                game_state,
                current,
                nearest_enemy.position,
                danger_map,
                allow_destruction=True
            )

            if path_with_destruction and len(path_with_destruction) > 1:
                next_pos = path_with_destruction[1]

                # Check if next position is blocked by a destructible block
                blocking_entity = get_blocking_entity_at(game_state, next_pos.x, next_pos.y)
                if blocking_entity is not None:
                    # Check if we already have a bomb targeting this
                    if has_bomb_targeting_position(game_state, unit, next_pos):
                        # Wait for bomb to clear the path
                        return None
                    # Try to bomb it
                    if should_place_bomb_to_clear(game_state, unit, next_pos, danger_map):
                        logger.info(f"Unit {unit_id} placing bomb to clear path (fallback)")
                        return BombAction(unit_id=unit_id)

        # Last resort: find any safe tile to move to
        safe_tile = find_nearest_safe_tile(game_state, current, danger_map)
        if safe_tile and (safe_tile.x != unit.x or safe_tile.y != unit.y):
            path = find_path(game_state, current, safe_tile, danger_map, allow_destruction=False)
            if path and len(path) > 1:
                next_pos = path[1]
                if self._is_move_valid(game_state, unit, next_pos):
                    direction = get_direction_to_point(current, next_pos)
                    if direction:
                        return MoveAction(unit_id=unit_id, move=direction)

        # If we can't move toward enemy or safe tile, try any valid adjacent tile
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_pos = Point(unit.x + dx, unit.y + dy)
            if self._is_move_valid(game_state, unit, next_pos):
                # Prefer non-dangerous tiles
                if not danger_map.is_dangerous(next_pos.x, next_pos.y):
                    direction = get_direction_to_point(current, next_pos)
                    if direction:
                        self._reserved_positions.add((next_pos.x, next_pos.y))
                        return MoveAction(unit_id=unit_id, move=direction)

        # Last resort: if completely stuck, try to bomb any adjacent block to create space
        from strategy import find_escape_creating_bomb
        if find_escape_creating_bomb(game_state, unit, danger_map):
            logger.info(f"Unit {unit_id} placing bomb to create escape route (stuck fallback)")
            return BombAction(unit_id=unit_id)

        return None

    async def _send_action(self, action: ActionPacket) -> None:
        """Send an action to the game server.

        Args:
            action: Action to send.
        """
        if isinstance(action, SkipAction):
            return  # Don't send skip actions

        await self._client._send(action.to_dict())


def main():
    """Main entry point."""
    logger.info("Starting AgentEx...")

    for attempt in range(10):
        try:
            Agent()
            break
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
