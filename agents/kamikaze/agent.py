# ruff: noqa: F405
import asyncio
import os
import time

from game_state import GameState as _GameState
from types_ import *  # pyright: ignore[reportAssignmentType]  # noqa: F403
from utils import is_any_enemy_in_my_armed_blast_radius, shortest_path_to_enemy

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
        self.unit_bombs: dict[UnitState, List[Point]] = {}

        self._client.set_game_tick_callback(self._on_game_tick)

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [
            asyncio.ensure_future(self._client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))


    async def _on_game_tick(self, tick_number, game_state):
        game_state = GameState(**game_state)
        
        if tick_number == 0:
            self.units_next_actions = {my_unit: [] for my_unit in game_state.my_units}
            self.unit_bombs = {my_unit: [] for my_unit in game_state.my_units}

        my_units = game_state.my_alive_units
        enemy_units = game_state.enemy_alive_units
        
        my_units_kills_enemy_unit: dict[UnitState, UnitState] = {my_unit: enemy_units[i % len(enemy_units)] for i, my_unit in enumerate(my_units)}

        for unit in my_units:
            unit_id = unit.unit_id
            
            # if an enemy can be killed, kill the fucker
            enemy_to_kill, bomb_position = is_any_enemy_in_my_armed_blast_radius(game_state, armed_ticks=5)
            if enemy_to_kill is not None:
                assert bomb_position is not None, "If there is an enemy to kill, bomb_position should be returned to know which bomb to denotate."
                await self._client.send_detonate(bomb_position.x, bomb_position.y, unit_id)
            
            # if there are any premoves, execute them
            elif len(self.units_next_actions[unit]) > 0:
                action_packet = self.units_next_actions[unit].pop(0)
                await self._client._send(action_packet)
            
            else:
                path = shortest_path_to_enemy(
                    game_state,
                    unit,
                    my_units_kills_enemy_unit[unit],
                )
                
                # place a bomb if we are next to the enemy
                if path is not None and len(path) <= 2:
                    # and the next moves should be to get to safety
                    await self._client.send_bomb(unit_id)
                
                # if the next step is safe, and able to move there (not blocked), move there
                elif path is not None and len(path) > 1:
                    next_point = path[1]
                    if game_state.is_walkable(next_point.x, next_point.y):
                        await self._client.send_move(
                            next_point.x,
                            next_point.y,
                            unit_id,
                        )


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
