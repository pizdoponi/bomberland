import asyncio
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from dqn_config import DQNConfig
from dqn_model import NUM_ACTIONS, ActionType, DQNModel
from dqn_shared import DQNFeatureBuilder, action_to_move
from game_state import GameState
from types_ import GameState as TypedGameState

logging.basicConfig(
    level=logging.INFO,
    format="[dqn] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

uri = os.environ.get(
    "GAME_CONNECTION_STRING"
) or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"


class DQNAgent:
    def __init__(self) -> None:
        self._client = GameState(uri)
        self._client.set_game_tick_callback(self._on_game_tick)

        self.config = DQNConfig.from_env()
        self._device = torch.device(self.config.device)

        self._feature_builder = DQNFeatureBuilder(self.config)

        self._model = None
        self._step_count = 0

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [asyncio.ensure_future(self._client._handle_messages(connection))]
        loop.run_until_complete(asyncio.wait(tasks))

    async def _on_game_tick(self, tick_number: int, game_state: Dict):
        self._step_count += 1

        typed_state = TypedGameState.from_dict(game_state)
        my_units = typed_state.my_units
        my_units_sorted = sorted([unit.unit_id for unit in my_units])

        if self._model is None:
            in_channels = self._feature_builder.channels * self.config.frame_stack_size
            self._model = DQNModel(
                conv_in_channels=in_channels,
                height=typed_state.world.height,
                width=typed_state.world.width,
                num_heads=self._feature_builder.num_heads,
                num_actions=NUM_ACTIONS,
                hidden_dim=self.config.hidden_dim,
            ).to(self._device)
            if os.path.exists(self.config.load_path):
                self._model.load(self.config.load_path)
                logger.info("Loaded checkpoint from %s", self.config.load_path)
            else:
                logger.warning(
                    "No checkpoint found at %s; using untrained model.",
                    self.config.load_path,
                )

        frame = self._feature_builder.encode_frame(
            typed_state, typed_state.my_agent_id
        )
        stacked_state = self._feature_builder.update_frame_stack(frame)
        cache = self._feature_builder.build_cache(typed_state)

        state_tensor = (
            torch.from_numpy(stacked_state).float().unsqueeze(0).to(self._device)
        )
        with torch.no_grad():
            q_values = self._model(state_tensor)[0].cpu().numpy()

        for unit_id in my_units_sorted:
            unit_state = typed_state.get_unit(unit_id)
            if unit_state is None or not unit_state.is_alive():
                continue

            head_index = self._feature_builder.unit_to_head_index(
                unit_id, my_units_sorted
            )
            if head_index is None:
                continue

            legal_actions = self._feature_builder.legal_actions(
                typed_state, unit_id, cache
            )
            action_index = self._select_action(q_values[head_index], legal_actions)
            action_type = ActionType.from_index(action_index)
            await self._execute_action(unit_id, action_type, cache.team_bombs)

    def _select_action(self, q_values: np.ndarray, legal_actions: List[int]) -> int:
        if not legal_actions:
            return ActionType.WAIT.value
        return max(legal_actions, key=lambda idx: q_values[idx])

    async def _execute_action(
        self, unit_id: str, action_type: ActionType, team_bombs: List[Tuple[int, int]]
    ) -> None:
        move = action_to_move(action_type)

        if move is not None:
            await self._client.send_move(move, unit_id)
        elif action_type == ActionType.PLACE_BOMB:
            await self._client.send_bomb(unit_id)
        elif action_type == ActionType.DETONATE_BOMB:
            if team_bombs:
                x, y = team_bombs[0]
                await self._client.send_detonate(x, y, unit_id)
        elif action_type == ActionType.WAIT:
            return
        else:
            logger.warning("Unhandled action %s for unit %s", action_type, unit_id)


def main() -> None:
    for _ in range(0, 10):
        while True:
            try:
                DQNAgent()
            except Exception as exc:
                logger.error("Agent error: %s", exc)
                time.sleep(5)
                continue
            break


if __name__ == "__main__":
    main()
