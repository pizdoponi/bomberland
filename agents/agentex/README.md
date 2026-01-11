# AgentEx - Rule-Based Bomberland Agent

## Overview

AgentEx is a rule-based agent for Bomberland that prioritizes safety while pursuing strategic objectives. It uses a layered decision-making architecture that adapts to different game phases.

## Architecture

```
agentex/
    agent.py          # Main agent class and game loop
    strategy.py       # High-level decision making and target assignment
    danger.py         # Danger zone calculation and escape route planning
    pathfinding.py    # Danger-aware A* pathfinding
    bomb_logic.py     # Bomb placement evaluation and detonation timing
    endgame.py        # Fire tracking and endgame positioning
    utils.py          # Helper functions (distance, direction, entity queries)
    types_.py         # Type definitions for game state, actions, entities
    game_state.py     # WebSocket client for game server communication
```

## Decision Priority System

Each tick, the agent evaluates actions for each unit in strict priority order:

### Priority 1: Escape Danger (Highest)
If a unit is in a danger zone (bomb blast radius, active blast, or fire):
1. Use BFS to find the shortest path to any safe tile
2. Move along the escape path immediately
3. If no escape exists, attempt to detonate a bomb (take enemy down with us)

### Priority 2: Detonate for Kill
Check if any armed bomb owned by this unit can hit an enemy:
- Verify enemy is in blast zone and not invulnerable
- Verify unit is NOT in blast zone (or is invulnerable)
- Verify no friendly units would be hit
- If all conditions met, detonate immediately

### Priority 3: Endgame Survival (Tick 200+)
When fire phase begins:
- Prioritize moving toward center/away from fire
- Calculate fire arrival times for each tile
- Path toward tiles with maximum time until fire arrives

### Priority 4: Strategic Actions
Use pre-computed decisions from the StrategyManager:
- **Move**: Follow path toward assigned target
- **Bomb**: Place bomb if escape path exists and valuable target in blast zone
- **Wait**: Hold position if blocked or no valid action

### Priority 5: Fallback
If no strategic decision available:
1. Try to pathfind toward nearest enemy
2. If blocked by destructible block, bomb to clear path
3. Find any safe tile to move toward
4. Move to any valid adjacent tile

## Danger Zone System (`danger.py`)

### DangerMap Class
Pre-computes all danger zones for the current tick:

**Tracked Dangers:**
- Active blasts (immediate danger)
- Bomb blast zones (considers blast radius and direction)
- Chain detonations (bombs triggering other bombs)
- Endgame fire (permanent, no expiration)

**Danger Timing:**
```python
DangerInfo:
    danger_start_tick  # When bomb becomes armed (created + 5 ticks)
    danger_end_tick    # When blast clears (explode + 5 ticks)
```

**Danger Levels:**
- 0: Safe tile
- 50: In bomb zone, bomb not yet armed
- 100: In armed bomb zone
- 1000 (INF): Active blast or fire

### Escape Route Calculation
```python
can_escape_after_bomb(unit, bomb_position, max_escape_moves=3):
    # BFS from unit position
    # Must reach tile outside BOTH:
    #   - The new bomb's hypothetical blast zone
    #   - Any existing danger zones
    # Within max_escape_moves (default 3)
```

## Pathfinding System (`pathfinding.py`)

### Danger-Aware A*
```python
Cost(tile) = base_cost + danger_penalty + destruction_cost

base_cost = 1 (empty tile)
danger_penalty = danger_level (0, 50, 100, or INF)
destruction_cost = 6 * block_hp (if allow_destruction=True)
    # wood = 1 HP, ore = 3 HP
    # 6 accounts for: place bomb + retreat + arm time + detonate + return
```

**Parameters:**
- `avoid_units`: Skip tiles with other units
- `avoid_danger`: Apply danger penalties
- `allow_destruction`: Include paths through destructible blocks
- `max_cost`: Limit maximum acceptable path cost
- `excluded_positions`: Additional tiles to avoid

### Key Functions
- `find_path()`: A* from start to specific goal
- `find_path_to_any()`: Find shortest path to any goal in a list
- `find_escape_path()`: BFS to find shortest path to ANY safe tile
- `find_safe_tiles()`: Get all safe reachable tiles within distance

## Strategy Manager (`strategy.py`)

### Game Phases
```python
EARLY_GAME:   tick < 100   # Focus on powerups
MID_GAME:     100 <= tick < 200  # Balanced offense
ENDGAME:      tick >= 200  # Survival priority
```

### Target Evaluation

**Powerup Scoring:**
```python
base_value:
    BLAST_POWERUP = 100
    FREEZE_POWERUP = 60

score = base_value / (path_cost + 1) * reachability_factor

reachability_factor:
    1.0 if safe path
    0.5 if path through danger
    0.0 if powerup expires before reachable

Early game multiplier: 1.5x
```

**Enemy Scoring:**
```python
base_value = 80 + (30 * missing_hp)  # Wounded enemies more valuable

score = base_value / (path_cost + 1) * path_type_factor

path_type_factor:
    1.0 for direct walkable path
    0.8 for path requiring block destruction

Mid/Endgame multiplier: 1.3x
```

### Target Assignment
Greedy assignment algorithm:
1. Collect all (unit, target, value) tuples
2. Sort by value (highest first)
3. Assign targets avoiding duplicates for non-enemy targets
4. Multiple units can target same enemy

### Unit Decisions
Decision types:
- `escape`: Move along escape path (highest priority)
- `move`: Move toward assigned target
- `bomb`: Place bomb at current position
- `detonate`: Detonate specific bomb
- `wait`: Do nothing this tick

## Bomb Logic (`bomb_logic.py`)

### Bomb Placement Conditions
A bomb is placed only when ALL conditions are true:
1. Agent has <3 active bombs (engine limit)
2. No bomb already at unit's position
3. Escape path exists (≤3 moves to safety)
4. No friendly units in blast zone
5. Something valuable in blast zone (enemy or destructible block)

### Blast Calculation
```python
blast_radius = (unit.blast_diameter - 1) // 2

# Blast extends in 4 cardinal directions
# Stops when hitting any block (including destructible)
# Default diameter = 3 (radius = 1)
```

### Detonation Timing
**Immediate Detonation Triggers:**
- Enemy enters armed bomb's blast zone
- Unit is safe from own blast
- No friendly units in blast zone

**Bomb Timeline:**
- Created: bomb placed
- Armed: created + 5 ticks (can be detonated)
- Auto-explode: created + 30 ticks (expires)
- Blast duration: 5 ticks

## Endgame Fire Handling (`endgame.py`)

### Fire Spiral Pattern
```
Fire starts: tick 200
Spawn interval: every 2 ticks
Pattern: Spirals inward from corners toward center (7,7)
    - Starts at edges (layer 0)
    - Each layer moves inward
    - Horizontal edges first, then vertical
```

### Survival Strategy
```python
should_prioritize_survival():
    if tick < 200: return False
    if safe_tiles < 50: return True
    if any unit has fire arriving in ≤5 ticks: return True
```

### Target Position Selection
Tiles scored by:
1. Maximum time until fire arrives (primary)
2. Distance to center (tiebreaker - closer is better)

## Unit Coordination

### Position Reservation
- Track intended positions for all friendly units
- If two units want same tile, lower-priority unit waits
- Priority order: escape > attack > collect > explore

### Collision Avoidance
- Skip positions occupied by other units in pathfinding
- Reserved positions blocked from subsequent decisions
- Units process in priority order each tick

## Key Constants

```python
# Timing
BOMB_ARMED_TICKS = 5
BOMB_DURATION_TICKS = 30
BLAST_DURATION_TICKS = 5
ENDGAME_START_TICK = 200
FIRE_SPAWN_INTERVAL = 2

# Limits
MAX_BOMBS_PER_AGENT = 3

# Pathfinding costs
COST_MOVE_EMPTY = 1
COST_DESTROY_PER_HP = 6
COST_DANGER_ARMED = 100
COST_DANGER_UNARMED = 50
COST_DANGER_ACTIVE = infinity

# Strategy weights
POWERUP_VALUE_BLAST = 100
POWERUP_VALUE_FREEZE = 60
ENEMY_VALUE_BASE = 80
ENEMY_VALUE_PER_MISSING_HP = 30
EARLY_GAME_END_TICK = 100
```

## Running the Agent

```bash
# Using docker-compose from project root
docker-compose up --abort-on-container-exit --force-recreate

# Direct Python execution (for development)
cd agents/agentex
pip install -r requirements.txt
python agent.py

# With custom connection string
GAME_CONNECTION_STRING="ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=agentex" python agent.py
```

## Dependencies

- Python 3.8+
- websockets (async WebSocket client)
- No external ML libraries (pure rule-based)
