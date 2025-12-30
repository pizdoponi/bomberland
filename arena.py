import argparse
import asyncio
import datetime as dt
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import dotenv
from tqdm import tqdm

dotenv.load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent
BASE_COMPOSE = ROOT_DIR / "base-compose.yml"
REPLAYS_DIR = ROOT_DIR / "replays"
DEFAULT_ADMIN_WS = "ws://127.0.0.1:3000/?role=admin"
REPLAY_PATH = ROOT_DIR / "agents" / "replay.json"


def run_command(
    command: list[str],
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT_DIR,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def parse_base_compose() -> Dict[str, Dict[str, Optional[str]]]:
    services: Dict[str, Dict[str, Optional[str]]] = {}
    current_service: Optional[str] = None
    in_services = False
    for raw_line in BASE_COMPOSE.read_text().splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "services:":
            in_services = True
            continue
        if not in_services:
            continue
        if line.startswith("    ") and stripped.endswith(":") and ":" not in stripped[:-1]:
            service_name = stripped[:-1]
            current_service = service_name
            services[current_service] = {"context": None}
            continue
        if current_service and stripped.startswith("context:"):
            context = stripped.split("context:", 1)[1].strip()
            services[current_service]["context"] = context
    return services


def resolve_agent_service(
    agent_name: str,
    services: Dict[str, Dict[str, Optional[str]]],
) -> str:
    if agent_name in services:
        return agent_name
    matches: list[str] = []
    for service_name, info in services.items():
        context = info.get("context")
        if not context:
            continue
        context_path = Path(context).as_posix().rstrip("/")
        if context_path.endswith(f"agents/{agent_name}"):
            matches.append(service_name)
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Unknown agent '{agent_name}'. Expected a service from base-compose.yml "
        "or a folder name under agents/."
    )


def generate_compose(
    agent_a_service: str,
    agent_b_service: str,
    agent_a_name: str,
    agent_b_name: str,
) -> str:
    return f"""version: "3"
services:
    game-engine:
        extends:
            file: base-compose.yml
            service: game-engine
        ports:
            - 3000:3000
        environment:
            - ADMIN_ROLE_ENABLED=1
            - AGENT_ID_MAPPING=agentA,agentB
            - INITIAL_HP=3
            - PRNG_SEED=1234
            - SHUTDOWN_ON_GAME_END_ENABLED=0
            - TELEMETRY_ENABLED=1
            - TICK_RATE_HZ=10
            - TRAINING_MODE_ENABLED=0
            - WORLD_SEED=9999
            - SAVE_REPLAY_ENABLED=1
        deploy:
            resources:
                limits:
                    cpus: "1"
                    memory: "1024M"
        networks:
            - coderone-tournament

    agent-a:
        extends:
            file: base-compose.yml
            service: {agent_a_service}
        environment:
            - GAME_CONNECTION_STRING=ws://game-engine:3000/?role=agent&agentId=agentA&name={agent_a_name}
        depends_on:
            - game-engine
        deploy:
            resources:
                limits:
                    cpus: "1"
                    memory: "1024M"
        networks:
            - coderone-tournament

    agent-b:
        extends:
            file: base-compose.yml
            service: {agent_b_service}
        environment:
            - GAME_CONNECTION_STRING=ws://game-engine:3000/?role=agent&agentId=agentB&name={agent_b_name}
        depends_on:
            - game-engine
        deploy:
            resources:
                limits:
                    cpus: "1"
                    memory: "1024M"
        networks:
            - coderone-tournament

networks:
    coderone-tournament:
"""


async def connect_admin(url: str) -> Any:
    try:
        import websockets
    except ImportError:
        raise RuntimeError(
            "Missing dependency: websockets. Install with `pip install websockets`."
        )
    return await websockets.connect(url)


async def wait_for_endgame(ws: Any, timeout_s: int) -> Dict[str, Any]:
    end_time = time.time() + timeout_s
    while True:
        remaining = end_time - time.time()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for endgame_state.")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        data = json.loads(raw)
        if data.get("type") == "endgame_state":
            return data.get("payload") or {}


def wait_for_replay_update(previous_mtime: float, timeout_s: int = 10) -> bool:
    end_time = time.time() + timeout_s
    while time.time() < end_time:
        if REPLAY_PATH.exists():
            mtime = REPLAY_PATH.stat().st_mtime
            if mtime > previous_mtime:
                return True
        time.sleep(0.2)
    return False


def save_logs(compose_path: Path, since_ts: str, output_path: Path) -> None:
    command = [
        "docker",
        "compose",
        "-f",
        str(compose_path),
        "logs",
        "--since",
        since_ts,
        "game-engine",
        "agent-a",
        "agent-b",
    ]
    result = run_command(command, check=False, capture_output=True)
    output_path.write_text(result.stdout or "")


async def run_arena(agent_a: str, agent_b: str, num_runs: int) -> None:
    services = parse_base_compose()
    agent_a_service = resolve_agent_service(agent_a, services)
    agent_b_service = resolve_agent_service(agent_b, services)

    compose_path = ROOT_DIR / ".arena-compose.yml"
    compose_path.write_text(
        generate_compose(agent_a_service, agent_b_service, agent_a, agent_b)
    )

    try:
        run_command(
            [
                "docker",
                "compose",
                "-f",
                str(compose_path),
                "up",
                "-d",
                "--build",
                "game-engine",
                "agent-a",
                "agent-b",
            ]
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to start docker compose services.") from exc

    time.sleep(5)

    try:
        ws = await connect_admin(DEFAULT_ADMIN_WS)
    except Exception as exc:
        print(
            "Warning: failed to connect as admin. Ensure ADMIN_ROLE_ENABLED=1 and the engine is reachable.",
            file=sys.stderr,
        )
        raise

    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for _ in tqdm(range(num_runs), desc="Arena runs"):
            world_seed = random.randint(1, 1_000_000)
            prng_seed = world_seed
            previous_mtime = REPLAY_PATH.stat().st_mtime if REPLAY_PATH.exists() else 0
            since_ts = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            reset_packet = {
                "type": "request_game_reset",
                "world_seed": world_seed,
                "prng_seed": prng_seed,
            }
            await ws.send(json.dumps(reset_packet))
            await wait_for_endgame(ws, timeout_s=600)

            replay_ready = wait_for_replay_update(previous_mtime)
            run_dir = REPLAYS_DIR / f"{agent_a}_{agent_b}_{world_seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if replay_ready and REPLAY_PATH.exists():
                replay_target = run_dir / "replay.json"
                replay_target.write_text(REPLAY_PATH.read_text())
            else:
                print(
                    f"Warning: replay.json not updated for seed {world_seed}.",
                    file=sys.stderr,
                )
            save_logs(compose_path, since_ts, run_dir / "logs.log")
    finally:
        await ws.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated Bomberland arena matches between two agents."
    )
    parser.add_argument("agent_a", help="Agent A name (base-compose.yml service or agents/ folder)")
    parser.add_argument("agent_b", help="Agent B name (base-compose.yml service or agents/ folder)")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs to simulate")
    args = parser.parse_args()

    if args.num_runs < 1:
        print("num-runs must be >= 1", file=sys.stderr)
        sys.exit(1)

    if not BASE_COMPOSE.exists():
        print("base-compose.yml not found in project root.", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(run_arena(args.agent_a, args.agent_b, args.num_runs))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
