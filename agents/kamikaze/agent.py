# ruff: noqa: F405
import asyncio
import logging
import os
import time

from game_state import GameState as _GameState
from types_ import *  # pyright: ignore[reportAssignmentType]  # noqa: F403
from utils import (
    is_any_enemy_in_my_armed_blast_radius,
    shortest_path_to_enemy,
)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[kamikaze] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

uri = os.environ.get(
    'GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"

actions = ["up", "down", "left", "right", "bomb", "detonate"]


class Agent():
    def __init__(self):
        self._client = _GameState(uri)

        # any initialization code can go here

        # premove for each unit if desired
        self.units_next_actions: dict[UnitState, List[ActionPacket]] = {}
        # keep track of which unit placed which bomb
        self.units_bombs: dict[UnitState, List[Point]] = {}
        # track movement history for retreating after bombing
        self.units_move_history: dict[UnitState, List[Point]] = {}

        self._client.set_game_tick_callback(self._on_game_tick)

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [
            asyncio.ensure_future(self._client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))

    def retreat_and_detonate(self, unit: UnitState):
        history = self.units_move_history[unit]
        retreat_moves: List[ActionPacket] = []
        current_point = unit.position
        for prev_point in list(reversed(history))[:3]:  # retreat three steps
            retreat_move = MoveAction.from_points(
                unit.unit_id, current_point, prev_point
            )
            if retreat_move is not None:
                retreat_moves.append(retreat_move)
                current_point = prev_point

        self.units_next_actions[unit] = retreat_moves
        self.units_next_actions[unit].append(
            DetonateAction(
                unit_id=unit.unit_id,
                target=unit.position,
            )
        )

    async def _on_game_tick(self, tick_number, game_state):
        game_state = GameState.from_dict(game_state)
        
        if tick_number == 0:
            self.units_next_actions = {my_unit: [] for my_unit in game_state.my_units}
            self.units_bombs = {my_unit: [] for my_unit in game_state.my_units}
            self.units_move_history = {my_unit: [] for my_unit in game_state.my_units}

        my_units = game_state.my_alive_units
        enemy_units = game_state.enemy_alive_units

        # which enemy unit each of my units is targeting (round robin)
        my_units_enemy_targets: dict[UnitState, UnitState] = {
            my_unit: enemy_units[i % len(enemy_units)]
            for i, my_unit in enumerate(my_units)
        }

        for unit in my_units:
            unit_id = unit.unit_id

            # if an enemy can be killed, kill the fucker
            enemy_to_kill, bomb_position = is_any_enemy_in_my_armed_blast_radius(
                game_state, armed_ticks=5
            )
            if enemy_to_kill is not None:
                assert bomb_position is not None, (
                    "If there is an enemy to kill, bomb_position should be returned to know which bomb to denotate."
                )
                # but only if the bomb belongs to this unit, we don't want to steal kills
                # the other unit will detonate its bomb
                if bomb_position in self.units_bombs[unit]:
                    # reset next moves
                    self.units_next_actions[unit] = []
                    await self._client.send_detonate(
                        bomb_position.x, bomb_position.y, unit_id
                    )
                    # proceed to the next unit
                    continue

            # if there are any premoves, execute them
            if len(self.units_next_actions[unit]) > 0:
                action_packet = self.units_next_actions[unit].pop(0)
                logger.info(f"Unit {unit.unit_id} executing premove: {action_packet}")
                await self._client._send(action_packet)

            # if the bomb is already placed, skip. Wait for detonation
            elif len(self.units_bombs[unit]) > 0:
                logger.info(f"Unit {unit.unit_id} waiting for detonation.")
                continue

            else:
                # find shortest path to the targeted enemy
                logger.info(
                    f"Unit {unit.unit_id} searching path to enemy {my_units_enemy_targets[unit].unit_id}."
                )
                path = shortest_path_to_enemy(
                    game_state,
                    unit,
                    my_units_enemy_targets[unit],
                )
                logger.info(f"Unit {unit.unit_id} path to enemy: {path}")

                # place a bomb if we are next to the enemy
                if path is not None and len(path) <= 2:
                    logger.info(
                        f"Unit {unit.unit_id} placing bomb to attack enemy {my_units_enemy_targets[unit].unit_id}."
                    )
                    # place bomb
                    await self._client.send_bomb(unit_id)
                    # retreat and detonate
                    self.retreat_and_detonate(unit)

                # if the next step is safe, and able to move there (not blocked), move there
                elif path is not None and len(path) > 1:
                    next_point = path[1]

                    if game_state.is_walkable(next_point.x, next_point.y):
                        next_move = MoveAction.from_points(
                            unit_id, unit.position, next_point
                        )
                        if next_move is not None:
                            logger.info(
                                f"Unit {unit.unit_id} moving towards enemy {my_units_enemy_targets[unit].unit_id} to {next_point}."
                            )
                            self.units_move_history[unit].append(unit.position)
                            await self._client._send(next_move)

                    else:
                        # place bomb, retreat, boom
                        logger.info(
                            f"Unit {unit.unit_id} cannot move to {next_point} as it is not walkable. Placing bomb and retreating."
                        )
                        await self._client.send_bomb(unit_id)
                        self.retreat_and_detonate(unit)

                else:
                    # no path to enemy found, do nothing this tick; should not happen
                    print(f"No path to enemy found for unit {unit.unit_id}.")
                    pass


def main():
    for i in range(0,10):
        while True:
            try:
                Agent()
            except Exception:
                time.sleep(5)
                continue
            break


if __name__ == "__main__":
    main()
