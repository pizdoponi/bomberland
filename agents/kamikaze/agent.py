# ruff: noqa: F405
import asyncio
import logging
import os
import time

from game_state import GameState as _GameState
from types_ import *  # pyright: ignore[reportAssignmentType]  # noqa: F403
from utils import (
    get_enemy_targets_for_my_units,
    get_retreat_path_after_bomb_placement,
    is_enemy_unit_in_my_units_armed_bomb_radius,
    shortest_path,
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

    def retreat_and_detonate(
        self,
        game_state: GameState,
        unit: UnitState,
        armed_ticks: int = 5,
        blast_duration: int = 5,
    ):
        retreat_path = list(reversed(self.units_move_history[unit]))

        if len(retreat_path) < (unit.blast_diameter // 2) + 1:
            logger.warning(
                f"Unit {unit.unit_id} does not have enough movement history to retreat safely after placing a bomb at {unit.position}. Current history length: {len(retreat_path)}, required: {(unit.blast_diameter // 2) + 1}. Finding alternative retreat path."
            )
            retreat_path = get_retreat_path_after_bomb_placement(game_state, unit)

            if retreat_path is None or (
                len(retreat_path) < (unit.blast_diameter // 2) + 1
            ):
                logger.error(
                    f"Unit {unit.unit_id} could not find a retreat path after placing a bomb at {unit.position}, or the retreat path is to short. Staying in place and detonating bomb."
                )
                for _ in range(armed_ticks):
                    self.units_next_actions[unit].append(
                        SkipAction(unit_id=unit.unit_id)
                    )
                self.units_next_actions[unit].append(
                    DetonateAction(
                        unit_id=unit.unit_id,
                        target=unit.position,
                    )
                )
                return

            retreat_path = retreat_path[: (unit.blast_diameter // 2) + 1]
            logger.info(
                f"Unit {unit.unit_id} found alternative retreat path to {retreat_path[-1]}: {retreat_path}"
            )
        else:
            retreat_path = retreat_path[: (unit.blast_diameter // 2) + 1]
            logger.info(
                f"Unit {unit.unit_id} retreating along movement history to {retreat_path[-1]}: {retreat_path}"
            )

        retreat_moves: List[ActionPacket] = []
        current_point = unit.position
        for next_point in retreat_path:
            retreat_move = MoveAction.from_points(
                unit.unit_id, current_point, next_point
            )
            if retreat_move is not None:
                retreat_moves.append(retreat_move)
                current_point = next_point

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
        # after detonation, wait for the blast to be over before pursuing the enemy again
        for _ in range(blast_duration):
            self.units_next_actions[unit].append(SkipAction(unit_id=unit.unit_id))

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
        my_units_enemy_targets = get_enemy_targets_for_my_units(
            game_state, my_units, enemy_units
        )
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
                    path = shortest_path(
                        game_state,
                        unit.position,
                        my_units_enemy_targets[unit].position,
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
                        self.retreat_and_detonate(game_state, unit)

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
                                # only append the move if it is different from the last position
                                # may happen if two units try to move to the same position, and block each other
                                if (
                                    len(self.units_move_history[unit]) == 0
                                    or self.units_move_history[unit][-1] != next_point
                                ):
                                    self.units_move_history[unit].append(unit.position)

                                await self._client._send(next_move.to_dict())

                        else:
                            # place bomb, retreat, boom
                            logger.info(
                                f"Unit {unit.unit_id} cannot move to {next_point} from {unit.position} as it is not walkable. Placing bomb at {unit.position} and retreating."
                            )
                            await self._client.send_bomb(unit_id)
                            self.retreat_and_detonate(game_state, unit)

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
