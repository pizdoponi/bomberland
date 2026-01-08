import asyncio
import logging
import os
import random
import time

from game_state import GameState
from types_ import ActionType, UnitState
from types_ import GameState as TypedGameState

uri = (
    os.environ.get("GAME_CONNECTION_STRING")
    or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Agent:
    def __init__(self):
        self._client = GameState(uri)

        # any initialization code can go here
        self._client.set_game_tick_callback(self._on_game_tick)

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [
            asyncio.ensure_future(self._client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))

    def get_random_legal_action(
        self, game_state: TypedGameState, unit: UnitState
    ) -> ActionType:
        all_legal_actions = game_state.legal_actions(unit)
        # only consider movements (and noop) to prevent self-destruction
        desired_legal_actions = [
            action
            for action in all_legal_actions
            if action
            in {
                ActionType.NOOP,
                ActionType.UP,
                ActionType.DOWN,
                ActionType.LEFT,
                ActionType.RIGHT,
            }
        ]
        assert len(desired_legal_actions) > 0, (
            "There should always be a legal action, NOOP at least."
        )
        return random.choice(desired_legal_actions)

    async def _on_game_tick(self, tick_number, _game_state):
        game_state = TypedGameState.from_dict(_game_state)

        # send each unit a random action
        for unit in game_state.my_alive_units:
            action = self.get_random_legal_action(game_state, unit)
            logger.info(
                f"Tick {tick_number}: Unit {unit.unit_id} takes action {action}"
            )

            if action == ActionType.NOOP:
                continue  # do nothing
            else:
                action_packet = action.to_action_packet(unit.unit_id, game_state)
                await self._client._send(action_packet.to_dict())


def main():
    for i in range(0, 10):
        while True:
            try:
                Agent()
            except:
                time.sleep(5)
                continue
            break


if __name__ == "__main__":
    main()
