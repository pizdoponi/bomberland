# Overview

`agent.py` - random agent

`agent_fwd.py` - random agent that connects to forward model

`dev_gym.py` - [open ai gym wrapper](https://gym.openai.com/)

## Agent Logic

A simple kamikaze agent. To keep it simple, each unit operates independently, following the logic below:

- Each unit is tasked of eliminating a corresponding enemy unit (unit 0 goes after enemy unit 0, etc.).
- Take the shortest path to the enemy unit, destroying obstacles in the way by placing bombs.
- Once in blast radius of the enemy unit, place a bomb and move away to avoid self-destruction (or not). KABOOM.
