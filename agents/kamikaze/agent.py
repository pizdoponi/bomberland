# ruff: noqa: F405
import asyncio
import logging
import os
import time

from game_state import GameState as _GameState
from types_ import *  # pyright: ignore[reportAssignmentType]  # noqa: F403
from utils import (
    is_enemy_unit_in_my_units_armed_bomb_radius,
    shortest_path_to_enemy,
)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[kamikaze] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

uri = (
    os.environ.get("GAME_CONNECTION_STRING")
    or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"
)

actions = ["up", "down", "left", "right", "bomb", "detonate"]


class Agent:
    def __init__(self):
        self._client = _GameState(uri)

        # premove for each unit if desired
        self.units_next_actions: dict[UnitState, List[ActionPacket]] = {}
        # keep track of which unit placed which bomb
        self.units_bombs: dict[UnitState, List[Point]] = {}
        # track movement history for retreating after bombing
        self.units_move_history: dict[UnitState, List[Point]] = {}

        # leave the rest to the async loop
        self._client.set_game_tick_callback(self._on_game_tick)
        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [
            asyncio.ensure_future(self._client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))

    def retreat_and_detonate(self, unit: UnitState, armed_ticks: int = 5):
        retreat_moves: List[ActionPacket] = []
        current_point = unit.position
        for prev_point in list(reversed(self.units_move_history[unit]))[
            :3
        ]:  # retreat three steps, if possible
            retreat_move = MoveAction.from_points(
                unit.unit_id, current_point, prev_point
            )
            if retreat_move is not None:
                retreat_moves.append(retreat_move)
                current_point = prev_point

        # set the premoves for this unit
        # first the retreat moves, to not be blown by own bomb
        self.units_next_actions[unit] = retreat_moves
        # wait for armed_ticks before detonating
        for _ in range(armed_ticks - len(retreat_moves)):
            self.units_next_actions[unit].append(SkipAction(unit_id=unit.unit_id))
        # finally, detonate at the bomb position
        self.units_next_actions[unit].append(
            DetonateAction(
                unit_id=unit.unit_id,
                target=unit.position,
            )
        )

    async def _on_game_tick(self, tick_number, game_state):
        game_state = GameState.from_dict(game_state)

        if tick_number == 1:
            logger.info("tick_number == 1, initializing unit data structures.")
            logger.info(f"My units: {game_state.my_units}")

            self.units_next_actions = {my_unit: [] for my_unit in game_state.my_units}
            self.units_bombs = {my_unit: [] for my_unit in game_state.my_units}
            self.units_move_history = {my_unit: [] for my_unit in game_state.my_units}

        my_units = game_state.my_alive_units
        enemy_units = game_state.enemy_alive_units

        # stupid check to avoid division by zero
        if len(enemy_units) == 0:
            logger.info("No enemy units alive, skipping turn.")
            return

        # which enemy unit each of my units is targeting (round robin)
        my_units_enemy_targets: dict[UnitState, UnitState] = {
            my_unit: enemy_units[i % len(enemy_units)]
            for i, my_unit in enumerate(my_units)
        }

        logger.info(
            f"On game tick {tick_number} my units are targeting: {',  '.join([my_unit.unit_id + '->' + my_units_enemy_targets[my_unit].unit_id for my_unit in my_units])}"
        )

        for unit in my_units:
            unit_id = unit.unit_id

            for enemy_unit in enemy_units:
                # if an enemy can be killed, kill the fucker
                is_enemy_killable, bomb_position = (
                    is_enemy_unit_in_my_units_armed_bomb_radius(
                        game_state, my_unit=unit, enemy_unit=enemy_unit, armed_ticks=5
                    )
                )
                if is_enemy_killable:
                    assert bomb_position is not None, (
                        "If there is an enemy to kill, bomb_position should be returned to know which bomb to denotate."
                    )
                    logger.info(
                        f"Unit {unit.unit_id} detonating bomb at {bomb_position} to kill enemy unit {enemy_unit.unit_id}"
                    )
                    # reset next moves
                    self.units_next_actions[unit] = []
                    await self._client.send_detonate(
                        bomb_position.x, bomb_position.y, unit_id
                    )
                    # proceed to the next unit
                    break
            # if we already detonated a bomb this tick, continue
            # else: (execute else block below)
            else:
                # if there are any premoves, execute them
                if len(self.units_next_actions[unit]) > 0:
                    action_packet = self.units_next_actions[unit].pop(0)
                    logger.info(
                        f"Unit {unit.unit_id} executing premove: {action_packet}. Next premoves: {self.units_next_actions[unit]}"
                    )
                    # send the action, and continue to next unit
                    # but only is not a SkipAction
                    if not isinstance(action_packet, SkipAction):
                        await self._client._send(action_packet.to_dict())

                # if the bomb is already placed, skip. Wait for detonation
                elif len(self.units_bombs[unit]) > 0:
                    logger.info(
                        f"Unit {unit.unit_id} waiting for detonation, skipping turn."
                    )
                    continue

                else:
                    # find shortest path to the targeted enemy
                    logger.info(
                        f"Unit {unit.unit_id} at {unit.position} searching path to enemy {my_units_enemy_targets[unit].unit_id} at {my_units_enemy_targets[unit].position}."
                    )
                    path = shortest_path_to_enemy(
                        game_state,
                        unit,
                        my_units_enemy_targets[unit],
                    )
                    logger.info(f"Unit {unit.unit_id} found path to enemy: {path}")

                    # place a bomb if we are next to the enemy
                    if path is not None and len(path) <= 1:
                        logger.info(
                            f"Unit {unit.unit_id} at {unit.position} placing bomb to attack enemy {my_units_enemy_targets[unit].unit_id} at {my_units_enemy_targets[unit].position}."
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
                                    f"Unit {unit.unit_id} at {unit.position} moving towards enemy {my_units_enemy_targets[unit].unit_id} at {my_units_enemy_targets[unit].position} by going to {next_point}."
                                )
                                self.units_move_history[unit].append(unit.position)
                                await self._client._send(next_move.to_dict())

                        else:
                            # place bomb, retreat, boom
                            logger.info(
                                f"Unit {unit.unit_id} cannot move to {next_point} from {unit.position} as it is not walkable. Placing bomb at {unit.position} and retreating."
                            )
                            await self._client.send_bomb(unit_id)
                            self.retreat_and_detonate(unit)

                    else:
                        # no path to enemy found, do nothing this tick; should not happen
                        logger.warning(
                            f"No path to enemy found for unit {unit.unit_id}."
                        )
                        pass


def main():
    for i in range(0, 10):
        while True:
            try:
                Agent()
            except Exception:
                time.sleep(5)
                continue
            break


if __name__ == "__main__":
    main()
