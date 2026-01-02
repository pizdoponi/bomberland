# Overview

`agent.py` - CNN-based DQN agent (PyTorch) for inference (loads a checkpoint and runs the policy)

`train.py` - training loop for the CNN DQN agent (replay buffer, target network, checkpointing)

`agent_fwd.py` - random agent that connects to forward model

`dev_gym.py` - [open ai gym wrapper](https://gym.openai.com/)

## DQN Usage

Training runs online and saves a checkpoint periodically. `agent.py` only runs inference.

CNN input uses a global board tensor with frame stacking. The network shares a CNN trunk and has three action heads, one per unit (sorted by unit ID).

Configuration lives in `DQNConfig` (`agents/dqn/config.py`). Environment variables are optional overrides:

- `DQN_EPSILON_START` (default: `1.0`)
- `DQN_EPSILON_MIN` (default: `0.1`)
- `DQN_EPSILON_DECAY` (default: `0.995`)
- `DQN_GAMMA` (default: `0.99`)
- `DQN_BATCH_SIZE` (default: `64`)
- `DQN_REPLAY_CAPACITY` (default: `20000`)
- `DQN_LEARNING_RATE` (default: `0.0005`)
- `DQN_HIDDEN_DIM` (default: `128`)
- `DQN_TARGET_UPDATE_INTERVAL` (default: `200`)
- `DQN_SAVE_INTERVAL` (default: `500`)
- `DQN_CHECKPOINT_PATH` (default: `checkpoints/dqn_cnn_weights.pt`)
- `DQN_LOAD_PATH` (default: `checkpoints/dqn_cnn_weights.pt`)
- `DQN_FRAME_STACK` (default: `4`)
- `DQN_MAX_ORE_HP` (default: `3`)
- `DQN_DEVICE` (default: `cpu`)

### Input channels

The per-frame tensor uses 8 channels in `C x H x W` order:

- `0` metal blocks
- `1` ore blocks (hp normalized)
- `2` wood blocks
- `3` bombs
- `4` blasts
- `5` my units
- `6` enemy units
- `7` powerups (ammo, blast power, freeze)
