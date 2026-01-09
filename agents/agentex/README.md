# AgentEx - Tournament-Competitive Rule-Based Bomberland Agent

## Overview

AgentEx is a sophisticated rule-based agent designed for tournament-level competition in Bomberland. It employs a multi-layered decision-making system that balances offensive and defensive play while adapting to different game phases.

## Strategy Summary

The agent follows a **Safety-First, Opportunity-Driven** approach:
- Never take actions that lead to certain death (unless sacrificing for a guaranteed kill)
- Pursue objectives (powerups, enemy kills) only when safe paths exist
- Adapt behavior based on game phase (early/mid/endgame)
- Coordinate multiple units to maximize map control and avoid friendly interference

## Detailed Strategy

### 1. Safety-First Decision Making

**Danger Zone Calculation:**
- Computes all tiles threatened by bombs (considering blast radius)
- Accounts for chain detonation (bombs triggering other bombs)
- Tracks bomb timers to know when danger zones will activate/clear
- Considers worst-case scenario (immediate detonation of armed bombs)

**Safety Priority:**
```
If unit is in danger zone:
    1. Find escape routes outside all blast radii
    2. Prioritize paths that become safe soonest
    3. If no escape: consider counter-attack or minimize damage
```

### 2. Three-Phase Game Strategy

#### Phase 1: Early Game (Ticks 0-100)
**Focus: Map Control & Powerup Collection**

- **Powerup Hunting**: Actively seek blast powerups (extends bomb range significantly)
- **Block Breaking**: Clear paths toward center and toward enemy spawn areas
- **Positioning**: Spread units to control different map quadrants
- **Conservative Play**: Avoid risky engagements; build advantage through powerups

*Motivation: Blast powerup gives +1 diameter per pickup. Early powerups compound into significant late-game advantage. A unit with diameter 7 vs diameter 3 controls 4x more area.*

#### Phase 2: Mid Game (Ticks 100-200)
**Focus: Balanced Aggression**

- **Opportunistic Attacks**: Engage enemies when favorable (near walls, limited escape)
- **Trap Setting**: Place bombs to limit enemy movement options
- **Territory Control**: Hold advantageous positions (center, near powerups)
- **Health Management**: Avoid trades when ahead in HP; force trades when behind

*Motivation: Mid-game is about converting early-game advantages into kills while maintaining safety margins before endgame pressure.*

#### Phase 3: Endgame (Ticks 200+)
**Focus: Survival & Positioning**

- **Center Positioning**: Move toward safe zones as fire spirals inward
- **Fire Awareness**: Pre-calculate fire spawn locations and timing
- **Pressure Tactics**: Use shrinking safe zone to force enemy into unfavorable positions
- **Last Stand Logic**: If cornered, maximize chance of taking enemies down

*Motivation: Fire spawns every 2 ticks from corners toward center. Units caught at edges are eliminated. Center control wins games.*

### 3. Bomb Placement Logic

**Place Bomb When:**
1. Enemy is within blast range AND cannot escape before detonation
2. Blocking enemy escape route (trap completion)
3. Breaking block that leads to valuable powerup
4. Breaking block to create path toward enemy
5. Endgame: Creating safe zones / blocking enemy paths

**Never Place Bomb When:**
1. No clear escape route for own unit
2. Would block own team's critical path
3. Enemy has easy escape and own unit is exposed
4. Already at max concurrent bombs (3 per agent)

**Retreat Planning:**
```
Before placing bomb:
    1. Calculate blast radius
    2. Verify at least one escape path exists
    3. Plan retreat moves (radius + 1 tiles minimum)
    4. Queue: [place_bomb, retreat_move_1, ..., retreat_move_n, wait_for_arm, detonate]
```

### 4. Detonation Strategy

**Immediate Detonation Triggers:**
- Enemy unit enters blast zone of armed bomb
- Chain reaction opportunity (own bomb triggers enemy bomb hitting enemy)
- Clearing path urgently needed (incoming fire/danger)

**Delayed Detonation:**
- Wait for enemy to move into blast zone
- Time with enemy's movement patterns
- Coordinate with other unit's bomb for crossfire

### 5. Pathfinding System

**Danger-Aware A* Search:**
```python
Cost(tile) = base_cost + danger_penalty + destruction_cost

base_cost = 1 (empty) or INF (solid block)
danger_penalty =
    - 0: safe tile
    - 100: in bomb blast zone (armed)
    - 50: in bomb blast zone (not yet armed)
    - INF: active blast/fire
destruction_cost =
    - 6 * HP: for destructible blocks (wood=1, ore=3)
    - Accounts for: bomb placement + retreat + detonate + return
```

### 6. Enemy Targeting

**Target Assignment Algorithm:**
1. Calculate shortest paths from each of my units to each enemy unit
2. Use Hungarian algorithm-style matching to minimize total distance
3. Prefer wounded enemies (lower HP = easier elimination)
4. Redistribute targets if an enemy is eliminated

**Engagement Rules:**
- Minimum 2-tile approach before bombing (prevents mutual destruction)
- Prefer attacking enemies near walls/corners (limited escape)
- Avoid chasing enemies through narrow corridors (trap risk)

### 7. Unit Coordination

**Collision Avoidance:**
- Track intended next positions for all friendly units
- If two units want same tile, lower-priority unit waits or reroutes
- Priority: escaping danger > completing attack > collecting powerup > exploring

**Spatial Distribution:**
- Penalize paths that cluster units together
- Ideal: triangular formation covering max area
- Benefit: harder for enemy to hit multiple units with one bomb

### 8. Powerup Evaluation

**Powerup Value Scoring:**
```
score = base_value / (distance + 1) * reachability_factor

base_value:
    - Blast Powerup: 100 (permanent advantage)
    - Freeze Powerup: 60 (temporary but powerful)

reachability_factor:
    - 1.0: clear safe path
    - 0.5: path through danger zones
    - 0.0: powerup expires before reachable
```

### 9. Chain Detonation Exploitation

**Offensive Chains:**
- Place bombs adjacent to enemy bombs
- Detonate own bomb to trigger enemy bomb
- Can extend effective blast range significantly

**Defensive Awareness:**
- Track all bombs that could trigger own bombs
- Factor chain reactions into danger zone calculations
- Avoid positions where enemy can trigger chain hitting own unit

### 10. Endgame Fire Handling

**Fire Spiral Pattern:**
```
Fire spawns at ticks: 200, 202, 204, ... (every 2 ticks)
Pattern: Starts top-left and bottom-right corners
        Spirals horizontally first, then vertically
        Converges on center (7,7)
```

**Survival Strategy:**
- Maintain positions at least 2 moves ahead of fire line
- Calculate fire arrival times for each tile
- Prefer center-adjacent positions as fire closes in
- Use fire to pressure enemies into bombs or each other

## Architecture

```
agentex/
    agent.py          # Main agent logic and game loop
    strategy.py       # High-level decision making
    danger.py         # Danger zone calculation
    pathfinding.py    # A* with danger awareness
    targeting.py      # Enemy targeting and assignment
    bomb_logic.py     # Bomb placement and detonation
    endgame.py        # Fire tracking and endgame behavior
    utils.py          # Helper functions
```

## Key Improvements Over Kamikaze Agent

| Aspect | Kamikaze | AgentEx |
|--------|----------|---------|
| Safety | Basic retreat path | Full danger zone modeling |
| Powerups | Ignored | Priority collection |
| Pathfinding | Simple Dijkstra | Danger-aware A* |
| Endgame | No adaptation | Fire-aware positioning |
| Coordination | Independent units | Coordinated targeting |
| Bomb Timing | Immediate detonate | Strategic timing |

## Performance Expectations

- **vs Random Agent**: ~99% win rate
- **vs Kamikaze Agent**: ~80%+ win rate
- **vs Similar Rule-Based**: Competitive (depends on specific strategies)
- **vs RL Agents**: Competitive in many cases (rule-based can exploit known patterns)

## Configuration

The agent's behavior can be tuned via constants in `strategy.py`:
- `EARLY_GAME_END_TICK`: When to transition from early to mid game
- `ENDGAME_START_TICK`: When fire phase begins
- `DANGER_PENALTY_ARMED`: Cost for entering armed bomb zone
- `POWERUP_VALUE_BLAST`: Base value of blast powerups
- `MIN_ESCAPE_DISTANCE`: Minimum retreat distance before bombing

## Running the Agent

```bash
# Using docker-compose from project root
docker-compose up --abort-on-container-exit --force-recreate

# Direct Python execution (for development)
cd agents/agentex
python agent.py
```

## Dependencies

- Python 3.8+
- websockets
- No external ML libraries required (pure rule-based)
