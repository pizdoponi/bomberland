import asyncio
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from dqn_config import DQNConfig
from dqn_model import NUM_ACTIONS, ActionType, DQNModel
from dqn_shared import DQNFeatureBuilder
from game_state import GameState
from types_ import GameState as TypedGameState
from types_ import MAX_CONCURRENT_BOMBS_PER_AGENT, SkipAction

load_dotenv()


logging.basicConfig(
    level=logging.DEBUG,
    format="[dqn-agent] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


agent_uri = (
    os.environ.get("GAME_CONNECTION_STRING")
    or "ws://game-engine:3000/?role=agent&agentId=agentA&name=dqn-agent"
)

logger.info(f"{agent_uri=}")


class DQNAgent:
    def __init__(self) -> None:
        self.agent_client = GameState(agent_uri)

        # Use default config for inference
        self.config = DQNConfig()
        logger.info(f"config={self.config}")

        self._feature_builder = DQNFeatureBuilder(self.config)

        self._model: Optional[DQNModel] = None
        self._step_count = 0

    async def run(self):
        logger.info(f"Connection agent to game engine at {agent_uri}")
        agent_connection = await self.agent_client.connect()

        self.agent_client.set_game_tick_callback(self._on_game_tick)

        agent_task = asyncio.create_task(
            self.agent_client._handle_messages(agent_connection)  # type: ignore
        )

        logger.info("Creating agent task")
        await asyncio.gather(agent_task)

    async def _on_game_tick(self, tick_number: int, game_state_: Dict):
        logger.debug(f"Step={self._step_count}, {tick_number=}")
        self._step_count += 1

        game_state = TypedGameState.from_dict(game_state_)

        my_units_sorted = sorted([unit.unit_id for unit in game_state.my_units])
        logger.debug(f"{my_units_sorted=}")

        if self._model is None:
            in_channels = (
                self._feature_builder.num_channels * self.config.frame_stack_size
            )
            self._model = DQNModel(
                conv_in_channels=in_channels,
                conv_hidden_channels=self.config.conv_hidden_channels,
                conv_out_channels=self.config.conv_out_channels,
                height=game_state.world.height,
                width=game_state.world.width,
                num_heads=self._feature_builder.num_heads,
                num_actions=NUM_ACTIONS,
                fc_hidden_dim=self.config.fc_hidden_dim,
            ).to(self.config.device)
            if not os.path.exists(self.config.load_path):
                raise FileNotFoundError(
                    f"Expected model checkpoint at {self.config.load_path}"
                )
            # Load checkpoint - handle both old and new format
            checkpoint = torch.load(
                self.config.load_path,
                map_location=self.config.device,
                weights_only=True,
            )
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from checkpoint (step {checkpoint.get('step_count', '?')})")
            else:
                self._model.load_state_dict(checkpoint)
                logger.info("Loaded model weights (legacy format)")
            self._model.eval()

        frame = self._feature_builder.encode_frame(game_state)
        stacked_state = self._feature_builder.update_frame_stack(frame)

        logger.debug(
            f"frame.shape={frame.shape}, stacked_state.shape={stacked_state.shape}"
        )

        state_tensor = (
            torch.from_numpy(stacked_state).float().unsqueeze(0).to(self.config.device)
        )
        with torch.no_grad():
            q_values = self._model(state_tensor)[0].cpu().numpy()

        agent_bombs_in_play = len(game_state.my_units_bombs())
        pending_bomb_placements = 0

        for unit_id in my_units_sorted:
            unit_state = game_state.get_unit(unit_id)
            if unit_state is None or not unit_state.is_alive():
                continue

            head_index = self._feature_builder.unit_id_to_head_index(
                unit_id, my_units_sorted
            )
            logger.debug(f"{unit_id=}, {head_index=}")
            if head_index is None:
                continue

            legal_action_types = game_state.legal_actions(unit_state)
            if (
                ActionType.PLACE_BOMB in legal_action_types
                and agent_bombs_in_play + pending_bomb_placements
                >= MAX_CONCURRENT_BOMBS_PER_AGENT
            ):
                legal_action_types = [
                    action
                    for action in legal_action_types
                    if action != ActionType.PLACE_BOMB
                ]
            legal_actions = [action.value for action in legal_action_types]

            action_index = self._select_action(q_values[head_index], legal_actions)
            action_type = ActionType.from_index(action_index)

            if action_type == ActionType.PLACE_BOMB:
                pending_bomb_placements += 1

            await self._execute_action(unit_id, action_type, game_state)

    def _select_action(self, q_values: np.ndarray, legal_actions: List[int]) -> int:
        if not legal_actions:
            logger.warning("No legal actions available, defaulting to NOOP")
            return ActionType.NOOP.value
        logger.debug(f"Selecting greedy action from {legal_actions=}")
        return max(legal_actions, key=lambda idx: q_values[idx])

    async def _execute_action(
        self, unit_id: str, action_type: ActionType, game_state: TypedGameState
    ) -> None:
        logger.debug(f"Executing action {action_type} for unit {unit_id}")
        action_packet = action_type.to_action_packet(unit_id, game_state)
        if isinstance(action_packet, SkipAction):
            logger.debug(f"Skipping action for unit {unit_id}, because {action_type=}")
            return
        else:
            logger.debug(f"Sending action packet for {action_type=} for unit {unit_id}")
        await self.agent_client._send(action_packet.to_dict())

def main() -> None:
    for _ in range(0, 10):
        while True:
            try:
                agent = DQNAgent()
                asyncio.run(agent.run())

            except Exception as exc:
                logger.error("Agent error: %s", exc)
                time.sleep(5)
                continue
            break


if __name__ == "__main__":
    main()
