import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from types_ import EntityType, GameState, Point, UnitState


def direction_from_to(current: Point, nxt: Point) -> Optional[str]:
    """
    Return a move direction ("up", "down", "left", "right") from current to nxt,
    or None if they are not 4-neighbour adjacent.
    """
    dx = nxt.x - current.x
    dy = nxt.y - current.y
    if dx == 1 and dy == 0:
        return "right"
    if dx == -1 and dy == 0:
        return "left"
    if dx == 0 and dy == 1:
        return "down"
    if dx == 0 and dy == -1:
        return "up"
    return None


def shortest_path_to_safe_square_after_bomb(
    game_state: GameState,
    my_unit: UnitState,
    bomb_position: Point,
    blast_diameter: Optional[int] = None,
) -> Optional[List[Point]]:
    """
    Find the shortest path to a walkable, safe tile after placing a bomb.

    A tile is considered safe if it is outside the bomb's blast radius and
    does not contain obviously dangerous entities (bombs/blasts).
    """
    start = my_unit.position

    if blast_diameter is None:
        blast_diameter = my_unit.blast_diameter or 3

    width, height = game_state.world.width, game_state.world.height
    solid_blocks = {
        EntityType.METAL_BLOCK,
        EntityType.ORE_BLOCK,
        EntityType.WOOD_BLOCK,
    }

    # Pre-compute blocked unit positions (all units except my_unit)
    blocked_unit_positions = set()
    for u in game_state.all_units:
        if u.unit_id == my_unit.unit_id:
            continue
        if u.is_alive():
            blocked_unit_positions.add((u.x, u.y))

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    def bomb_blast_tiles() -> Set[Tuple[int, int]]:
        tiles: Set[Tuple[int, int]] = set()
        bx, by = bomb_position.x, bomb_position.y
        tiles.add((bx, by))

        radius = max(0, (blast_diameter - 1) // 2)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for step in range(1, radius + 1):
                x = bx + dx * step
                y = by + dy * step
                if not in_bounds(x, y):
                    break
                tiles.add((x, y))
                entities_here = game_state.entities_at(x, y)
                if any(e.entity_type in solid_blocks for e in entities_here):
                    break
        return tiles

    blast_tiles = bomb_blast_tiles()

    def is_safe(x: int, y: int) -> bool:
        return (x, y) not in blast_tiles and not game_state.is_dangerous_tile(x, y)

    def can_traverse(x: int, y: int) -> bool:
        if not in_bounds(x, y):
            return False
        if (x, y) in blocked_unit_positions:
            return False
        return game_state.is_walkable(x, y, ignore_bombs=True)

    # BFS for shortest path in number of moves
    start_key = (start.x, start.y)
    queue = deque([start_key])
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = {start_key}

    # Early exit if we're already safe
    if is_safe(start.x, start.y):
        return [start]

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    target_key: Optional[Tuple[int, int]] = None

    while queue:
        x, y = queue.popleft()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            nkey = (nx, ny)
            if nkey in visited:
                continue
            if not can_traverse(nx, ny):
                continue
            visited.add(nkey)
            parent[nkey] = (x, y)
            if is_safe(nx, ny):
                target_key = nkey
                queue.clear()
                break
            queue.append(nkey)

    if target_key is None:
        return None

    # Reconstruct path from target back to start
    path_coords: List[Tuple[int, int]] = []
    cur = target_key
    while cur != start_key:
        path_coords.append(cur)
        cur = parent[cur]
    path_coords.append(start_key)
    path_coords.reverse()

    return [Point(x, y) for (x, y) in path_coords]


def shortest_path_to_enemy(
    game_state: GameState,
    my_unit: UnitState,
    enemy_unit: UnitState,
) -> Optional[List[Point]]:
    """
    Compute the lowest-cost path from `my_unit` to `enemy_unit` using a
    weighted BFS (Dijkstra) over the grid.

    Movement model:
        - Moving into a free tile costs 1 (one game tick).
        - Moving into a tile containing destructible obstacles (wood / ore) costs:
              1 (move) + (1 bomb + 2 retreat + 1 detonate + 2 return) * hp
            = 1 + 6 * hp
        - Tiles with metal blocks are treated as impassable.
        - Tiles occupied by other units (besides the enemy target) are impassable.

    Returns:
        A list of Points from start to goal (inclusive), or None if unreachable.
    """
    start = my_unit.position
    goal = enemy_unit.position

    # Early exit
    if start.x == goal.x and start.y == goal.y:
        return [start]

    width, height = game_state.world.width, game_state.world.height

    # Pre-compute blocked unit positions (all units except my_unit & enemy_unit)
    blocked_unit_positions = set()
    for u in game_state.all_units:
        if u.unit_id in (my_unit.unit_id, enemy_unit.unit_id):
            continue
        if u.is_alive():
            blocked_unit_positions.add((u.x, u.y))

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    def tile_traversal_cost(x: int, y: int) -> Optional[int]:
        """
        Return the cost to enter tile (x, y), or None if it is impassable.
        """
        if not in_bounds(x, y):
            return None

        # Block other units (cannot walk through them)
        if (x, y) in blocked_unit_positions and not (x == goal.x and y == goal.y):
            return None

        entities = game_state.entities_at(x, y)

        # Hard-block: metal is not traversable
        has_metal = any(e.entity_type == EntityType.METAL_BLOCK for e in entities)
        if has_metal:
            return None

        # Sum HP of destructible obstacles (wood / ore) on this tile
        hp_total = 0
        for e in entities:
            if e.entity_type == EntityType.WOOD_BLOCK:
                hp_total += e.hp if e.hp is not None else 1
            elif e.entity_type == EntityType.ORE_BLOCK:
                hp_total += e.hp if e.hp is not None else 3

        base_cost = 1  # cost to move into an empty tile

        if hp_total > 0:
            extra = 6 * hp_total  # (1 bomb + 2 retreat + 1 detonate + 2 return) * hp
            return base_cost + extra

        return base_cost

    # Dijkstra / weighted BFS
    start_key = (start.x, start.y)
    goal_key = (goal.x, goal.y)

    # Priority queue entries: (total_cost_so_far, x, y)
    pq: List[Tuple[int, int, int]] = [(0, start.x, start.y)]
    heapq.heapify(pq)

    # Distance and parent maps
    dist: Dict[Tuple[int, int], int] = {start_key: 0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # 4-neighbour grid (cardinal moves only)
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while pq:
        cost, x, y = heapq.heappop(pq)
        key = (x, y)

        # Skip outdated queue entries
        if cost > dist.get(key, float("inf")):
            continue

        # Reached goal
        if key == goal_key:
            break

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            step_cost = tile_traversal_cost(nx, ny)
            if step_cost is None:
                continue

            new_cost = cost + step_cost
            nkey = (nx, ny)

            if new_cost < dist.get(nkey, float("inf")):
                dist[nkey] = new_cost
                parent[nkey] = key
                heapq.heappush(pq, (new_cost, nx, ny))

    # If goal is unreachable
    if goal_key not in dist:
        return None

    # Reconstruct path from goal back to start
    path_coords: List[Tuple[int, int]] = []
    cur = goal_key
    while cur != start_key:
        path_coords.append(cur)
        cur = parent[cur]
    path_coords.append(start_key)
    path_coords.reverse()

    # Convert to list[Point]
    return [Point(x, y) for (x, y) in path_coords]


def is_enemy_in_my_armed_blast_radius(
    game_state: GameState,
    enemy_unit: UnitState,
    armed_ticks: int = 5,
) -> Tuple[bool, Optional[Point]]:
    """
    Return True (and bomb position) if `enemy_unit` is in the blast radius of any bomb placed by
    *our* agent and that bomb has been on the map for at least `armed_ticks`.

    Rules used:
        - A bomb is "ours" if its owner_unit_id belongs to our agent's unit_ids.
        - A bomb is armable if: game_state.tick >= bomb.created + armed_ticks.
        - Blast shape: cross (up, down, left, right) with radius derived from
            bomb.blast_diameter (or owner unit's blast_diameter, or default=3).
        - Blast propagation stops when it hits a solid block:
            METAL_BLOCK, ORE_BLOCK, WOOD_BLOCK.
        - If the enemy stands on the bomb tile itself, they're in range.
    """
    tick = game_state.tick
    my_agent = game_state.my_agent
    my_unit_ids = set(my_agent.unit_ids)
    enemy_x, enemy_y = enemy_unit.x, enemy_unit.y

    # For convenience
    world = game_state.world

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < world.width and 0 <= y < world.height

    # Map from unit_id -> blast_diameter (for bombs missing blast_diameter)
    unit_blast_map = {u.unit_id: u.blast_diameter for u in game_state.all_units}

    # Get all bombs belonging to us
    my_bombs = [
        e
        for e in game_state.entities
        if e.entity_type == EntityType.BOMB and e.owner_unit_id in my_unit_ids
    ]

    for bomb in my_bombs:
        # Check if bomb is armed long enough to be remotely detonated
        if tick < bomb.created + armed_ticks:
            continue

        bx, by = bomb.x, bomb.y

        # If enemy stands on the bomb tile itself, it will be hit
        if enemy_x == bx and enemy_y == by:
            return True, Point(bx, by)

        # Determine blast diameter / radius
        if bomb.blast_diameter is not None:
            blast_diameter = bomb.blast_diameter
        else:
            # Fallback: use owning unit's blast_diameter or default to 3
            blast_diameter = unit_blast_map.get(bomb.owner_unit_id, 3)

        # Convert diameter (3,5,7,...) to radius (1,2,3,...)
        radius = max(0, (blast_diameter - 1) // 2)

        # Check each of the 4 cardinal directions
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for step in range(1, radius + 1):
                x = bx + dx * step
                y = by + dy * step

                if not in_bounds(x, y):
                    break

                # If we see a solid block, blast stops here
                entities_here = game_state.entities_at(x, y)
                if any(
                    e.entity_type
                    in {
                        EntityType.METAL_BLOCK,
                        EntityType.ORE_BLOCK,
                        EntityType.WOOD_BLOCK,
                    }
                    for e in entities_here
                ):
                    # The block itself is hit, but blast does not go past it
                    # (enemy can't stand on it anyway).
                    break

                # Check if enemy is on this tile
                if enemy_x == x and enemy_y == y:
                    return True, Point(bx, by)

    return False, None


def is_any_enemy_in_my_armed_blast_radius(
    game_state: GameState,
    armed_ticks: int = 5,
) -> Tuple[Optional[UnitState], Optional[Point]]:
    """
    Return any enemy unit if it is in the blast radius of any bomb placed by
    *our* agent and that bomb has been on the map for at least `armed_ticks`.
    """
    for enemy_unit in game_state.enemy_alive_units:
        is_enemy_killable, bomb_position = is_enemy_in_my_armed_blast_radius(
            game_state,
            enemy_unit,
            armed_ticks,
        )
        if is_enemy_killable:
            return enemy_unit, bomb_position # type: ignore
    return None, None
