import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from types_ import Entity, EntityType, GameState, Point, UnitState


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


def get_enemy_targets_for_my_units(
    game_state: GameState,
    my_units: List[UnitState],
    enemy_units: List[UnitState],
) -> Dict[UnitState, UnitState]:
    """
    Assign each of `my_units` to an enemy in `enemy_units` such that the assignments
    have the lowest total cost to reach the targets, where cost is defined as the length
    of the shortest path from my unit to the enemy unit.
    """
    costs: List[Tuple[int, UnitState, UnitState]] = []  # (cost, my_unit, enemy_unit)
    path_cache: Dict[
        Tuple[UnitState, UnitState], Optional[List[Point]]
    ] = {}  # (my_unit, enemy_unit) -> path
    for my_unit in my_units:
        for enemy_unit in enemy_units:
            path = shortest_path_to_enemy(game_state, my_unit, enemy_unit)
            path_cache[(my_unit, enemy_unit)] = path
            if path is not None:
                cost = len(path)
                costs.append((cost, my_unit, enemy_unit))
    # Sort costs ascending
    costs.sort(key=lambda x: x[0])
    # keep track of assigned units
    assigned_my_units: Set[UnitState] = set()
    assigned_enemy_units: Set[UnitState] = set()
    assignments: Dict[UnitState, UnitState] = {}

    for cost, my_unit, enemy_unit in costs:
        if my_unit in assigned_my_units or enemy_unit in assigned_enemy_units:
            continue
        assignments[my_unit] = enemy_unit
        assigned_my_units.add(my_unit)
        assigned_enemy_units.add(enemy_unit)

    # if some of my_units are unassigned, assign them randomly to the closest enemy
    unassigned_my_units = [u for u in my_units if u not in assigned_my_units]

    for my_unit in unassigned_my_units:
        closest_enemy = None
        closest_cost = float("inf")
        for enemy_unit in enemy_units:
            path = path_cache.get((my_unit, enemy_unit))
            if path is not None:
                cost = len(path)
                if cost < closest_cost:
                    closest_cost = cost
                    closest_enemy = enemy_unit
        if closest_enemy is not None:
            assignments[my_unit] = closest_enemy

    return assignments


def is_enemy_unit_in_my_units_armed_bomb_radius(
    game_state: GameState,
    my_unit: UnitState,
    enemy_unit: UnitState,
    armed_ticks: int = 5,
) -> Tuple[bool, Optional[Point]]:
    """
    Return True (and bomb position) if `enemy_unit` is in the blast radius of any bomb placed by
    `my_unit` and that bomb has been on the map for at least `armed_ticks`, and there is no
    obstacle between the bomb and the enemy unit.
    """

    def is_in_bounds(x: int, y: int) -> bool:
        """Return True if (x, y) is within world bounds."""
        return 0 <= x < game_state.world.width and 0 <= y < game_state.world.height

    def is_my_units_bomb(bomb: Entity) -> bool:
        """Return True if bomb belongs to `my_unit`."""
        return (
            bomb.entity_type == EntityType.BOMB
            and bomb.owner_unit_id == my_unit.unit_id
        )

    def is_bomb_armed(bomb: Entity) -> bool:
        """Return True if bomb has been on the map for at least `armed_ticks`."""
        assert bomb.entity_type == EntityType.BOMB, (
            "is_bomb_armed called on non-bomb entity."
        )
        return game_state.tick >= bomb.created + armed_ticks

    my_units_armed_bombs = [
        e for e in game_state.entities if is_my_units_bomb(e) and is_bomb_armed(e)
    ]

    for bomb in my_units_armed_bombs:
        blast_diameter = bomb.blast_diameter or my_unit.blast_diameter or 3
        radius = max(0, (blast_diameter - 1) // 2)

        # check in each direction, for up to `radius` tiles
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            for step in range(1, radius + 1):
                maybe_enemy_position = Point(bomb.x + dx * step, bomb.y + dy * step)

                if not is_in_bounds(maybe_enemy_position.x, maybe_enemy_position.y):
                    continue

                if bomb.position.distance_to(maybe_enemy_position) > radius:
                    continue

                if (
                    maybe_enemy_position.x == enemy_unit.x
                    and maybe_enemy_position.y == enemy_unit.y
                ):
                    # check for obstacles between bomb and enemy
                    is_blast_blocked = False
                    for obstacle_step in range(1, step):
                        ox = bomb.x + dx * obstacle_step
                        oy = bomb.y + dy * obstacle_step
                        entities_here = game_state.entities_at(ox, oy)
                        if any(
                            e.entity_type
                            in {
                                EntityType.METAL_BLOCK,
                                EntityType.ORE_BLOCK,
                                EntityType.WOOD_BLOCK,
                            }
                            for e in entities_here
                        ):
                            is_blast_blocked = True
                            break
                    if not is_blast_blocked:
                        return True, bomb.position

    return False, None  # Default return if no bomb hits
