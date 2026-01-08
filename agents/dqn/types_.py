from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants aligned with engine defaults.
BOMB_ARMED_TICKS = 5
MAX_CONCURRENT_BOMBS_PER_AGENT = 3

# mines


class ActionType(Enum):
    NOOP = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    PLACE_BOMB = 5
    DETONATE_BOMB_0 = 6
    DETONATE_BOMB_1 = 7
    DETONATE_BOMB_2 = 8

    @classmethod
    def ordered(cls) -> list["ActionType"]:
        return [
            cls.NOOP,
            cls.UP,
            cls.RIGHT,
            cls.DOWN,
            cls.LEFT,
            cls.PLACE_BOMB,
            cls.DETONATE_BOMB_0,
            cls.DETONATE_BOMB_1,
            cls.DETONATE_BOMB_2,
        ]

    @classmethod
    def from_index(cls, index: int) -> "ActionType":
        return cls(index)

    def to_action_packet(self, unit_id: str, game_state: "GameState") -> "ActionPacket":
        if self == ActionType.NOOP:
            return SkipAction(unit_id=unit_id)
        elif self == ActionType.UP:
            return MoveAction.from_direction(unit_id=unit_id, direction="up")
        elif self == ActionType.DOWN:
            return MoveAction.from_direction(unit_id=unit_id, direction="down")
        elif self == ActionType.LEFT:
            return MoveAction.from_direction(unit_id=unit_id, direction="left")
        elif self == ActionType.RIGHT:
            return MoveAction.from_direction(unit_id=unit_id, direction="right")
        elif self == ActionType.PLACE_BOMB:
            return BombAction(unit_id=unit_id)
        elif self == ActionType.DETONATE_BOMB_0:
            maybe_bomb_0 = game_state.my_units_bombs(unit_id=unit_id, bomb_idx=0)
            if not maybe_bomb_0:
                return SkipAction(unit_id=unit_id)
            bomb_0 = maybe_bomb_0[0]
            return DetonateAction(unit_id=unit_id, target=Point(bomb_0.x, bomb_0.y))
        elif self == ActionType.DETONATE_BOMB_1:
            maybe_bomb_1 = game_state.my_units_bombs(unit_id=unit_id, bomb_idx=1)
            if not maybe_bomb_1:
                return SkipAction(unit_id=unit_id)
            bomb_1 = maybe_bomb_1[0]
            return DetonateAction(unit_id=unit_id, target=Point(bomb_1.x, bomb_1.y))
        elif self == ActionType.DETONATE_BOMB_2:
            maybe_bomb_2 = game_state.my_units_bombs(unit_id=unit_id, bomb_idx=2)
            if not maybe_bomb_2:
                return SkipAction(unit_id=unit_id)
            bomb_2 = maybe_bomb_2[0]
            return DetonateAction(unit_id=unit_id, target=Point(bomb_2.x, bomb_2.y))
        else:
            raise ValueError(f"Unknown ActionType: {self}")

    def is_bomb_detonation(self) -> bool:
        return self in {
            ActionType.DETONATE_BOMB_0,
            ActionType.DETONATE_BOMB_1,
            ActionType.DETONATE_BOMB_2,
        }

    def is_movement(self) -> bool:
        return self in {
            ActionType.UP,
            ActionType.DOWN,
            ActionType.LEFT,
            ActionType.RIGHT,
        }


# ─────────────────────────────────────────────────────────────
# Basic enums & small helper types
# ─────────────────────────────────────────────────────────────


class AgentId(str, Enum):
    """Identifier for an agent in Bomberland.

    Official values:
        * "a" - Agent A
        * "b" - Agent B
    """

    A = "a"
    B = "b"


class Role(str, Enum):
    """Connection role used when talking to the game server."""

    AGENT = "agent"
    SPECTATOR = "spectator"
    ADMIN = "admin"


class MoveDirection(str, Enum):
    """Cardinal directions used in movement action packets."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class EntityType(str, Enum):
    """Type of an entity in the `entities` list.

    See the Bomberland docs "Game Entities" section.
    """

    AMMO = "a"  # Ammunition pickup
    BOMB = "b"  # Bomb
    BLAST = "x"  # Blast / end-game fire
    BLAST_POWERUP = "bp"  # Blast radius powerup
    FREEZE_POWERUP = "fp"  # Freeze powerup
    METAL_BLOCK = "m"  # Metal (indestructible)
    ORE_BLOCK = "o"  # Ore (3 HP)
    WOOD_BLOCK = "w"  # Wooden (1 HP)


@dataclass
class Point:
    """2D grid coordinate in the Bomberland world.

    Coordinates follow the engine convention: [x, y].
    """

    x: int
    y: int

    @classmethod
    def from_sequence(cls, seq: Iterable[Union[int, float]]) -> "Point":
        """Create a Point from a JSON coordinate list, e.g. [3, 10]."""
        x, y = list(seq)
        return cls(int(x), int(y))

    def as_list(self) -> List[int]:
        """Return coordinates in the JSON-compatible [x, y] format."""
        return [self.x, self.y]

    def distance_to(self, other: Point) -> int:
        """Compute Manhattan distance to another Point."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


# ─────────────────────────────────────────────────────────────
# Core game objects
# ─────────────────────────────────────────────────────────────


@dataclass
class Inventory:
    """Inventory for a unit.

    Attributes
    ----------
    bombs:
        Number of bombs available to place (ammunition).
        In Bomberland v4 this is effectively infinite, but the
        server still reports a value for compatibility.
    """

    bombs: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Inventory":
        return cls(bombs=int(data.get("bombs", 0)))


@dataclass(eq=False)
class UnitState:
    """State of a single unit (player-controlled character).

    This corresponds to one entry in the `unit_state` object.

    Attributes
    ----------
    unit_id:
        Unique identifier for this unit (e.g. "c", "d", "e"...).
    agent_id:
        ID of the owning agent ("a" or "b").
    position:
        Current location of the unit as a Point [x, y].
    hp:
        Current health points of the unit.
    inventory:
        Inventory data, currently only bombs.
    blast_diameter:
        Blast diameter for bombs placed by this unit.
    invulnerable_until:
        Latest tick number (inclusive) during which the unit
        is invulnerable. If current tick <= this value, the unit
        will not take damage.
    stunned_until:
        Latest tick number (inclusive) during which the unit
        is stunned. If current tick <= this value, the unit
        cannot move or perform actions.
    """

    unit_id: str
    agent_id: AgentId
    position: Point
    hp: int
    inventory: Inventory
    blast_diameter: int
    invulnerable_until: int
    stunned_until: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnitState":
        """Parse a `unit_state[unit_id]` JSON object into UnitState."""
        coords = data.get("coordinates") or data.get("coord") or [0, 0]
        position = Point.from_sequence(coords)

        inventory = Inventory.from_dict(data.get("inventory", {}))

        agent_raw = data.get("agent_id")
        if agent_raw is None:
            agent_raw = data.get("owner_id")  # used in some update payloads

        invulnerable_raw = data.get("invulnerable")
        if invulnerable_raw is None:
            invulnerable_raw = data.get(
                "invulnerability"
            )  # used in some update payloads

        unit_id_raw = data.get("unit_id")
        if unit_id_raw is None:
            unit_id_raw = data.get("id")  # some payloads use "id"

        return cls(
            unit_id=str(unit_id_raw),
            agent_id=AgentId(str(agent_raw)),
            position=position,
            hp=int(data.get("hp", 0)),
            inventory=inventory,
            blast_diameter=int(data.get("blast_diameter", 0)),
            invulnerable_until=int(invulnerable_raw or 0),
            stunned_until=int(data.get("stunned", 0)),
        )

    # Convenience helpers for agent code:

    @property
    def x(self) -> int:
        """Shortcut for self.position.x."""
        return self.position.x

    @property
    def y(self) -> int:
        """Shortcut for self.position.y."""
        return self.position.y

    def is_alive(self) -> bool:
        """Return True if the unit has at least 1 HP."""
        return self.hp > 0

    def is_invulnerable(self, tick: int) -> bool:
        """Return True if the unit is invulnerable at the given tick."""
        return tick <= self.invulnerable_until

    def is_stunned(self, tick: int) -> bool:
        """Return True if the unit is stunned at the given tick."""
        return tick <= self.stunned_until

    def __hash__(self):
        return hash(self.unit_id)

    def __eq__(self, other):
        if not isinstance(other, UnitState):
            return False
        return self.unit_id == other.unit_id


@dataclass
class Agent:
    """Agent-level information.

    This corresponds to each entry in the `agents` object.

    Attributes
    ----------
    agent_id:
        Agent identifier ("a" or "b").
    unit_ids:
        IDs of the units controlled by this agent.
    """

    agent_id: AgentId
    unit_ids: List[str]

    @classmethod
    def from_dict(cls, agent_id: str, data: Mapping[str, Any]) -> "Agent":
        return cls(
            agent_id=AgentId(agent_id),
            unit_ids=list(data.get("unit_ids", [])),
        )


@dataclass
class Entity:
    """Representation of a non-unit object on the map (`entities` array).

    Depending on the `type`, different fields may be present. All optional
    attributes default to None when not applicable.

    Attributes
    ----------
    created:
        Tick on which this entity was created (0 if part of the initial world).
    position:
        Location of the entity on the grid.
    entity_type:
        Type of entity (bomb, block, powerup, blast, etc.).
    expires:
        Tick on which this entity will disappear or explode
        (only for bombs, powerups, blasts, etc.).
    hp:
        Hit points remaining before the entity is destroyed
        (for destructible blocks, powerups, etc.).
    owner_unit_id:
        Unit that owns this entity (e.g. the unit that placed a bomb
        or whose blast this is). Some entities will not have an owner.
    blast_diameter:
        Blast diameter if this entity is a bomb.
    """

    created: int
    position: Point
    entity_type: EntityType
    expires: Optional[int] = None
    hp: Optional[int] = None
    owner_unit_id: Optional[str] = None
    blast_diameter: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Entity":
        """Parse an entry from the `entities` list."""
        position = Point(int(data.get("x", 0)), int(data.get("y", 0)))
        raw_type = str(data.get("type"))
        entity_type = EntityType(raw_type)

        # Historically this has been called both "owner_unit_id" and "unit_id"
        owner_unit_id = (
            data.get("owner_unit_id")
            or data.get("unit_id")  # fall back to older field name
        )

        expires = data.get("expires")
        hp = data.get("hp")
        blast_diameter = data.get("blast_diameter")

        return cls(
            created=int(data.get("created", 0)),
            position=position,
            entity_type=entity_type,
            expires=int(expires) if expires is not None else None,
            hp=int(hp) if hp is not None else None,
            owner_unit_id=str(owner_unit_id) if owner_unit_id is not None else None,
            blast_diameter=int(blast_diameter) if blast_diameter is not None else None,
        )

    # UX helpers for pathfinding / reasoning:

    @property
    def x(self) -> int:
        return self.position.x

    @property
    def y(self) -> int:
        return self.position.y

    def is_solid(self) -> bool:
        """Return True if units cannot move through this entity."""
        return self.entity_type in {
            EntityType.METAL_BLOCK,
            EntityType.ORE_BLOCK,
            EntityType.WOOD_BLOCK,
            EntityType.BOMB,
        }

    def is_pickup(self) -> bool:
        """Return True if this entity is a pickup (powerup/ammo)."""
        return self.entity_type in {
            EntityType.AMMO,
            EntityType.BLAST_POWERUP,
            EntityType.FREEZE_POWERUP,
        }

    def is_dangerous(self, current_tick: int) -> bool:
        """Return True if standing on this tile is immediately dangerous.

        This marks things like bombs and active blasts / end-game fire as
        dangerous, but you may want additional logic based on timers.
        """
        return self.entity_type == EntityType.BLAST or (
            self.entity_type == EntityType.BOMB and self.is_armed(current_tick)
        )

    def is_armed(self, current_tick: int) -> bool:
        """Return True if this entity is a bomb that is armed (for at least 5 ticks)."""
        if self.entity_type == EntityType.BOMB:
            return current_tick - self.created > BOMB_ARMED_TICKS
        return False

    def blast_diameter_(self, unit: Optional[UnitState] = None) -> int:
        """Return the blast diameter if this entity is a bomb.

        If the entity is not a bomb, returns 0. If a UnitState is provided
        and it matches the owner of the bomb, returns that unit's blast
        diameter (to account for powerups).
        """
        if self.entity_type != EntityType.BOMB:
            return 0
        if unit is not None and self.owner_unit_id == unit.unit_id:
            return unit.blast_diameter
        if self.blast_diameter is not None:
            return self.blast_diameter  # type: ignore
        return 3  # default bomb blast diameter

    def blast_radius(self, unit: Optional[UnitState] = None) -> int:
        """Return the blast radius if this entity is a bomb.

        If the entity is not a bomb, returns 0. If a UnitState is provided
        and it matches the owner of the bomb, returns that unit's blast
        radius (to account for powerups).
        """
        diameter = self.blast_diameter_(unit)
        return max(0, diameter // 2)

    def time_until_expires(self, current_tick: int) -> Optional[float]:
        """Return number of ticks until this entity expires.

        If the entity does not have an expiration tick, returns None.
        Normalized to the interval [0, 1] based on the lifetime of the entity type.
        A value of 1.0 means just created, 0.0 means about to expire.
        Value 0.0 means no entity (when creating a frame).
        """
        entity_type_lifetimes = {
            EntityType.BOMB: 30,
            EntityType.BLAST: 5,
            EntityType.BLAST_POWERUP: 40,
            EntityType.FREEZE_POWERUP: 40,
        }
        if self.expires is None:
            return None
        return max(
            0,
            (self.expires - current_tick)
            / entity_type_lifetimes.get(self.entity_type, 1),
        )


@dataclass
class World:
    """Static information about the world grid.

    Attributes
    ----------
    width:
        Number of cells horizontally (default is 15).
    height:
        Number of cells vertically (default is 15).
    """

    width: int
    height: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "World":
        return cls(
            width=int(data.get("width", 15)),
            height=int(data.get("height", 15)),
        )

    def in_bounds(self, point: Point) -> bool:
        """Return True if a point lies within the world boundaries."""
        return 0 <= point.x < self.width and 0 <= point.y < self.height


@dataclass
class Config:
    """Configuration settings for the game environment.

    These values come from the `config` object in `game_state`.
    """

    tick_rate_hz: int
    game_duration_ticks: int
    fire_spawn_interval_ticks: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        return cls(
            tick_rate_hz=int(data.get("tick_rate_hz", 10)),
            game_duration_ticks=int(data.get("game_duration_ticks", 1800)),
            fire_spawn_interval_ticks=int(data.get("fire_spawn_interval_ticks", 5)),
        )


@dataclass
class Connection:
    """Information about your agent's connection.

    Attributes
    ----------
    id:
        Connection ID used internally by tournament servers.
    role:
        Role of the connection (agent, spectator, admin).
    agent_id:
        Which logical agent this connection controls ("a" or "b").
    """

    id: int
    role: Role
    agent_id: AgentId

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Connection":
        return cls(
            id=int(data.get("id", 0)),
            role=Role(str(data.get("role", Role.AGENT.value))),
            agent_id=AgentId(str(data.get("agent_id", AgentId.A.value))),
        )


# ─────────────────────────────────────────────────────────────
# High-level GameState wrapper
# ─────────────────────────────────────────────────────────────


@dataclass
class GameState:
    """Typed wrapper around the Bomberland `game_state` JSON.

    This is meant to be the single entry point for your agent code.
    Use `GameState.from_dict(game_state)` each tick and then work with
    attributes and helper methods instead of manual JSON indexing.

    Attributes
    ----------
    agents:
        Mapping from AgentId -> Agent object.
    units:
        Mapping from unit_id -> UnitState.
    entities:
        List of all non-unit entities on the map.
    world:
        World dimensions.
    tick:
        Current game tick.
    config:
        Game configuration.
    connection:
        Information about your connection (e.g. which agent you are).
    """

    agents: Dict[AgentId, Agent]
    units: Dict[str, UnitState]
    entities: List[Entity]
    world: World
    tick: int
    config: Config
    connection: Connection
    _blast_tiles_cache: Dict[tuple, Set[Point]] = field(
        default_factory=dict, repr=False
    )
    # Spatial index: (x, y) -> list of entities at that position
    _entity_grid: Dict[tuple, List[Entity]] = field(
        default_factory=dict, repr=False
    )
    # Unit position index: (x, y) -> unit at that position
    _unit_grid: Dict[tuple, UnitState] = field(
        default_factory=dict, repr=False
    )
    # Bomb cache: owner_unit_id -> list of bombs owned by that unit
    _bombs_by_owner: Dict[str, List[Entity]] = field(
        default_factory=dict, repr=False
    )
    # All bombs cache
    _all_bombs: Optional[List[Entity]] = field(
        default=None, repr=False
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GameState":
        """Parse a raw `game_state` JSON object from the engine."""
        # Agents
        agents_raw = data.get("agents", {})
        agents: Dict[AgentId, Agent] = {}
        for agent_id_str, agent_data in agents_raw.items():
            agent_id = AgentId(str(agent_id_str))
            agents[agent_id] = Agent.from_dict(agent_id_str, agent_data)

        # Units
        units_raw = data.get("unit_state", {})
        units: Dict[str, UnitState] = {
            unit_id: UnitState.from_dict(unit_data)
            for unit_id, unit_data in units_raw.items()
        }

        # Entities
        entities_raw = data.get("entities", []) or []
        entities: List[Entity] = [Entity.from_dict(e) for e in entities_raw]

        world = World.from_dict(data.get("world", {}))
        config = Config.from_dict(data.get("config", {}))
        connection = Connection.from_dict(data.get("connection", {}))
        tick = int(data.get("tick", 0))

        # Build spatial index for O(1) entity lookups
        entity_grid: Dict[tuple, List[Entity]] = {}
        bombs_by_owner: Dict[str, List[Entity]] = {}
        all_bombs: List[Entity] = []
        for e in entities:
            key = (e.x, e.y)
            if key not in entity_grid:
                entity_grid[key] = []
            entity_grid[key].append(e)

            # Build bomb indices
            if e.entity_type == EntityType.BOMB:
                all_bombs.append(e)
                owner = e.owner_unit_id
                if owner is not None:
                    if owner not in bombs_by_owner:
                        bombs_by_owner[owner] = []
                    bombs_by_owner[owner].append(e)

        # Build unit position index
        unit_grid: Dict[tuple, UnitState] = {}
        for u in units.values():
            unit_grid[(u.x, u.y)] = u

        return cls(
            agents=agents,
            units=units,
            entities=entities,
            world=world,
            tick=tick,
            config=config,
            connection=connection,
            _entity_grid=entity_grid,
            _unit_grid=unit_grid,
            _bombs_by_owner=bombs_by_owner,
            _all_bombs=all_bombs,
        )

    # ------------- Agent-centric helpers -------------

    @property
    def my_agent_id(self) -> AgentId:
        """AgentId controlled by this connection."""
        return self.connection.agent_id

    @property
    def my_agent(self) -> Agent:
        """Agent object for the current connection."""
        return self.agents[self.my_agent_id]

    @property
    def enemy_agent_id(self) -> AgentId:
        """The other AgentId (assuming 2-player game)."""
        return AgentId.B if self.my_agent_id == AgentId.A else AgentId.A

    @property
    def enemy_agent(self) -> Agent:
        """Agent object for the opponent."""
        return self.agents[self.enemy_agent_id]

    @property
    def my_units(self) -> List[UnitState]:
        """List of unit states belonging to the current agent."""
        return [self.units[uid] for uid in self.my_agent.unit_ids if uid in self.units]

    @property
    def enemy_units(self) -> List[UnitState]:
        """List of unit states belonging to the opponent."""
        return [
            self.units[uid] for uid in self.enemy_agent.unit_ids if uid in self.units
        ]

    @property
    def all_units(self) -> List[UnitState]:
        """All known units (both agents)."""
        return list(self.units.values())

    @property
    def alive_units(self) -> List[UnitState]:
        """All units that currently have >0 HP."""
        return [u for u in self.all_units if u.is_alive()]

    @property
    def my_alive_units(self) -> List[UnitState]:
        """Units belonging to the current agent that are still alive."""
        return [u for u in self.my_units if u.is_alive()]

    @property
    def enemy_alive_units(self) -> List[UnitState]:
        """Units belonging to the opponent that are still alive."""
        return [u for u in self.enemy_units if u.is_alive()]

    def get_unit(self, unit_id: str) -> Optional[UnitState]:
        """Get a unit by ID, or None if it doesn't exist in this state."""
        return self.units.get(unit_id)

    def is_game_over(self) -> bool:
        """Return True if the game is over (one agent has no living units)."""
        my_alive = any(u.is_alive() for u in self.my_units)
        enemy_alive = any(u.is_alive() for u in self.enemy_units)
        return not (my_alive and enemy_alive)

    def my_units_bombs(
        self, unit_id: Optional[str] = None, bomb_idx: Optional[int] = None
    ) -> List[Entity]:
        """Returns bombs placed by my units.

        Args:
            unit_id: If provided, only returns bombs placed by this unit.
            bomb_idx: If provided, only returns the bomb at this index (0-based). Must be in range [0, 2],
                corresponding to the first, second, or third bomb placed by the unit (based on game_tick).
                Should only be used when `unit_id` is also provided.

        Returns:
            List of Entity objects corresponding to bombs placed by my units.
            The list is empty if no such bombs exist (if requesting bomb at idx 2 for unit that only placed
                one bomb, the return is also an empty list).

        Uses bomb cache for O(1) lookup per unit instead of O(n) entity iteration.
        """
        assert bomb_idx is None or bomb_idx in {0, 1, 2}, (
            "bomb_idx must be one of {0, 1, 2} if provided"
        )
        assert not (unit_id is None and bomb_idx is not None), (
            "bomb_idx can only be used when unit_id is provided"
        )

        # Use cached bomb lookup instead of iterating all entities
        if unit_id is not None:
            my_units_bombs = self._bombs_by_owner.get(unit_id, []).copy()
        else:
            my_unit_ids = {u.unit_id for u in self.my_units}
            my_units_bombs = []
            for uid in my_unit_ids:
                my_units_bombs.extend(self._bombs_by_owner.get(uid, []))

        if bomb_idx is not None:
            # Sort bombs by creation tick to determine order
            my_units_bombs.sort(key=lambda b: b.created)
            if my_units_bombs and len(my_units_bombs) > bomb_idx:
                my_units_bombs = [my_units_bombs[bomb_idx]]
            else:
                my_units_bombs = []

        logger.debug(f"My units' bombs for unit {unit_id} at index {bomb_idx} at tick {self.tick}: {my_units_bombs}")

        return my_units_bombs

    def legal_actions(self, unit: UnitState) -> List[ActionType]:
        """Return a list of legal ActionType values for the given unit."""
        actions = [ActionType.NOOP]

        execution_tick = self.tick + 1
        if unit.is_stunned(execution_tick):
            return [ActionType.NOOP]  # If stunned, only NOOP is legal

        # Movement
        for direction, (dx, dy) in {
            ActionType.UP: (0, 1),
            ActionType.DOWN: (0, -1),
            ActionType.LEFT: (-1, 0),
            ActionType.RIGHT: (1, 0),
        }.items():
            new_x = unit.x + dx
            new_y = unit.y + dy
            if self.is_walkable(new_x, new_y, ignore_units=False):
                actions.append(direction)

        # Bomb placement - use spatial index for O(1) check
        agent_unit_ids = {
            unit_state.unit_id
            for unit_state in self.units.values()
            if unit_state.agent_id == unit.agent_id
        }
        # Count agent bombs using cached bomb lookup - O(k) where k = num agents
        agent_bomb_count = sum(
            len(self._bombs_by_owner.get(uid, []))
            for uid in agent_unit_ids
        )
        # O(1) check if bomb already at unit's position
        entities_at_unit = self._entity_grid.get((unit.x, unit.y), [])
        bomb_already_placed_here = any(
            e.entity_type == EntityType.BOMB for e in entities_at_unit
        )
        if (
            unit.inventory.bombs > 0
            and agent_bomb_count < MAX_CONCURRENT_BOMBS_PER_AGENT
            and not bomb_already_placed_here
        ):
            actions.append(ActionType.PLACE_BOMB)

        # Bomb detonations
        my_bombs = sorted(
            self.my_units_bombs(unit_id=unit.unit_id), key=lambda bomb: bomb.created
        )
        if len(my_bombs) >= 1 and my_bombs[0].is_armed(execution_tick):
            actions.append(ActionType.DETONATE_BOMB_0)
        if len(my_bombs) >= 2 and my_bombs[1].is_armed(execution_tick):
            actions.append(ActionType.DETONATE_BOMB_1)
        if len(my_bombs) >= 3 and my_bombs[2].is_armed(execution_tick):
            actions.append(ActionType.DETONATE_BOMB_2)

        logger.debug(
            f"Legal actions for unit {unit.unit_id} at tick {self.tick} are: {[a.name for a in actions]}"
        )

        return actions

    def entities_of_type(self, entity_type: EntityType) -> List[Entity]:
        """Return all entities with the given EntityType."""
        return [e for e in self.entities if e.entity_type == entity_type]

    def entities_at(
        self,
        x: int,
        y: int,
        types: Optional[Iterable[EntityType]] = None,
    ) -> List[Entity]:
        """Return entities at a given coordinate.

        Parameters
        ----------
        x, y:
            Coordinate of interest.
        types:
            Optional iterable of EntityType values to filter by.

        Uses spatial index for O(1) lookup instead of O(n) iteration.
        """
        point = Point(x, y)
        if not self.world.in_bounds(point):
            return []

        result = self._entity_grid.get((x, y), [])
        if types is not None:
            type_set = set(types)
            result = [e for e in result if e.entity_type in type_set]
        return result

    def is_walkable(
        self,
        x: int,
        y: int,
        ignore_bombs: bool = False,
        ignore_units: bool = False,
    ) -> bool:
        """Return True if a unit can move into tile (x, y).

        This checks world bounds and solid entities. By default bombs
        are considered blocking. Set `ignore_bombs=True` if you want to
        consider bombs as walkable (e.g. for planning via bomb timing).
        Set `ignore_units=True` to ignore other units on the tile.

        Uses spatial indices for O(1) lookups.
        """
        point = Point(x, y)
        if not self.world.in_bounds(point):
            return False

        if not ignore_units:
            # O(1) lookup instead of iterating all units
            if (x, y) in self._unit_grid:
                return False

        entities_here = self._entity_grid.get((x, y), [])
        for e in entities_here:
            if e.entity_type in {
                EntityType.METAL_BLOCK,
                EntityType.ORE_BLOCK,
                EntityType.WOOD_BLOCK,
            }:
                return False
            if not ignore_bombs and e.entity_type == EntityType.BOMB:
                return False
        return True

    def is_dangerous_tile(self, x: int, y: int) -> bool:
        """Return True if the tile contains obviously dangerous entities.

        This is a simple helper that checks bombs and current blasts.
        You may want to extend this logic to look at timers when doing
        more advanced planning.

        Uses cached bomb list for O(k) instead of O(n) where k = number of bombs.
        """
        entities_here = self.entities_at(x, y)
        if any(e.is_dangerous(self.tick) for e in entities_here):
            return True

        # simulate bomb blasts using cached bomb list
        for bomb in self._all_bombs or []:
            blast_tiles = self.get_blast_tiles_if_detonated(bomb.position)
            if Point(x, y) in blast_tiles:
                return True

        return False

    def get_blast_tiles_if_detonated(
        self,
        position: Point,
        _visited: Optional[Set[Point]] = None,
        require_armed: bool = True,
    ) -> Set[Point]:
        """Return a set of Points that would be affected by a bomb blast.

        If there is no bomb at the given position, or (when require_armed is True)
        that bomb is not armed, returns empty list.
        Stops blast propagation when hitting solid blocks, and propagates
        through other bombs (including their blast tiles).

        Results are cached by position for efficiency within this GameState instance.
        """
        entities_here = self.entities_at(position.x, position.y)
        bombs_here = [e for e in entities_here if e.entity_type == EntityType.BOMB]

        # there should be at most one bomb at a given position
        assert len(bombs_here) <= 1, (
            f"Multiple bombs found at the same position {position}"
        )

        if len(bombs_here) == 0:
            return set()  # no bomb at this position

        bomb = bombs_here[0]

        if require_armed and not bomb.is_armed(self.tick):
            return set()  # bomb is not armed

        # Create cache key based on bomb position
        cache_key = (position.x, position.y, require_armed)

        # Check cache first
        if cache_key in self._blast_tiles_cache:
            return self._blast_tiles_cache[cache_key]

        # Track visited bombs to avoid infinite recursion in chain detonations
        if _visited is None:
            _visited = set()

        if position in _visited:
            return set()  # Already processing this bomb (cycle detection)

        _visited.add(position)

        # bomb's own tile is always blown up
        blast_tiles = {position}

        # get blast radius
        unit_id = bomb.owner_unit_id
        assert unit_id is not None, "Bomb should have an owner_unit_id"
        blast_radius = bomb.blast_radius(unit=self.get_unit(unit_id))

        # check in each cardinal direction
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            for distance in range(1, blast_radius + 1):
                nx = position.x + dx * distance
                ny = position.y + dy * distance
                next_point = Point(nx, ny)

                if not self.world.in_bounds(next_point):
                    break

                blast_tiles.add(next_point)

                # stop if we hit a solid entity
                entities_here = self.entities_at(nx, ny)
                if any(
                    e.entity_type
                    in {
                        EntityType.METAL_BLOCK,
                        EntityType.ORE_BLOCK,
                        EntityType.WOOD_BLOCK,
                    }
                    for e in entities_here
                ):
                    break

                # propagate the blast if a bomb is hit
                if any(e.entity_type == EntityType.BOMB for e in entities_here):
                    blast_tiles.update(
                        self.get_blast_tiles_if_detonated(
                            next_point,
                            _visited,
                            require_armed=False,
                        )
                    )
                    break

        # Cache the result
        self._blast_tiles_cache[cache_key] = blast_tiles

        logger.debug(
            f"Blast tiles for bomb at {position} (tick {self.tick}): {blast_tiles}"
        )

        return blast_tiles


# ─────────────────────────────────────────────────────────────
# Action packet dataclasses
# ─────────────────────────────────────────────────────────────


@dataclass
class ActionPacket:
    """Base class for all agent action packets.

    You normally won't instantiate this directly; instead use one of
    the concrete subclasses like BombAction, MoveAction, etc.

    Subclasses must implement `to_dict()` to produce a JSON packet
    that matches `ValidAgentPacket` in the engine schema.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict suitable for sending over the websocket."""
        raise NotImplementedError


@dataclass
class BombAction(ActionPacket):
    """Action: place a bomb with a specific unit.

    JSON form:
        {"type": "bomb", "unit_id": "<unit_id>"}
    """

    unit_id: str
    type: str = field(init=False, default="bomb")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "unit_id": self.unit_id,
        }


@dataclass
class MoveAction(ActionPacket):
    """Action: move a unit in a cardinal direction.

    JSON form:
        {"type": "move", "move": "up" | "down" | "left" | "right",
         "unit_id": "<unit_id>"}
    """

    unit_id: str
    move: MoveDirection
    type: str = field(init=False, default="move")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "move": self.move.value,
            "unit_id": self.unit_id,
        }

    @staticmethod
    def from_direction(unit_id: str, direction: str) -> "MoveAction":
        """Create a MoveAction from a string direction."""
        return MoveAction(unit_id=unit_id, move=MoveDirection(direction))

    @staticmethod
    def get_direction(
        unit_id: str, current: Point, next: Point
    ) -> Optional["MoveAction"]:
        """Create a MoveAction from current and next Point positions."""
        if next.x == current.x + 1 and next.y == current.y:
            return MoveAction(unit_id=unit_id, move=MoveDirection.RIGHT)
        elif next.x == current.x - 1 and next.y == current.y:
            return MoveAction(unit_id=unit_id, move=MoveDirection.LEFT)
        elif next.x == current.x and next.y == current.y + 1:
            return MoveAction(unit_id=unit_id, move=MoveDirection.UP)
        elif next.x == current.x and next.y == current.y - 1:
            return MoveAction(unit_id=unit_id, move=MoveDirection.DOWN)
        else:
            return None

    @staticmethod
    def from_points(
        unit_id: str, current: Point, next: Point
    ) -> Optional["MoveAction"]:
        """Create a MoveAction from current and next Point positions."""
        direction = MoveAction.get_direction(unit_id, current, next)
        if direction is None:
            return None
        return MoveAction(unit_id=unit_id, move=direction.move)


@dataclass
class DetonateAction(ActionPacket):
    """Action: remotely detonate a bomb at given coordinates.

    JSON form:
        {"type": "detonate", "coordinates": [x, y], "unit_id": "<unit_id>"}
    """

    unit_id: str
    target: Point
    type: str = field(init=False, default="detonate")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "coordinates": self.target.as_list(),
            "unit_id": self.unit_id,
        }


@dataclass
class RequestTickAction(ActionPacket):
    """Admin-only action: request the next tick in training mode.

    JSON form:
        {"type": "request_tick"}
    """

    type: str = field(init=False, default="request_tick")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type}


@dataclass
class SkipAction(ActionPacket):
    """Action: skip the turn for a unit (do nothing).

    JSON form:
        {"type": "skip", "unit_id": "<unit_id>"}

    #! WARNING: this is NOT an officially supported action in Bomberland!
    It should thus never be sent to the server, it is only used internally
    by agents to represent a no-op for a unit.
    """

    unit_id: str
    type: str = field(init=False, default="skip")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "unit_id": self.unit_id,
        }


# ─────────────────────────────────────────────────────────────
# Small top-level helper
# ─────────────────────────────────────────────────────────────


def parse_game_state(raw: Mapping[str, Any]) -> GameState:
    """Convenience function to parse a raw `game_state` dict.

    Example
    -------
    >>> state = parse_game_state(game_state)
    >>> for unit in state.my_units:
    ...     print(unit.unit_id, unit.position, unit.hp)
    """
    return GameState.from_dict(raw)
