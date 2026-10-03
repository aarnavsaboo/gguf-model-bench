from pathlib import Path
import json

from .planner import BenchJob


def read_jobs(path: str) -> list[BenchJob]:
    return [BenchJob(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def read_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path: str, rows: list[dict]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
