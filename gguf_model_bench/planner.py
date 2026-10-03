from dataclasses import asdict, dataclass
from hashlib import sha1
from itertools import product
import json


@dataclass(frozen=True)
class BenchJob:
    id: str
    model: str
    prompt_tokens: int
    generation_tokens: int
    depth: int
    batch_size: int
    ubatch_size: int
    cache_k: str
    cache_v: str
    threads: int | None
    gpu_layers: int
    repetitions: int

    def to_dict(self):
        return asdict(self)


def _list(value):
    return value if isinstance(value, list) else [value]


def expand(config: dict) -> list[BenchJob]:
    dimensions = product(
        _list(config["models"]),
        _list(config.get("prompt_tokens", [512])),
        _list(config.get("generation_tokens", [128])),
        _list(config.get("depth", [0])),
        _list(config.get("batch_size", [2048])),
        _list(config.get("ubatch_size", [512])),
        _list(config.get("cache_k", ["f16"])),
        _list(config.get("cache_v", ["f16"])),
        _list(config.get("threads", [None])),
        _list(config.get("gpu_layers", [-1])),
    )
    rows = []
    for values in dimensions:
        payload = dict(zip(
            ["model","prompt_tokens","generation_tokens","depth","batch_size","ubatch_size","cache_k","cache_v","threads","gpu_layers"],
            values,
        ))
        payload["repetitions"] = int(config.get("repetitions", 5))
        ident = sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        rows.append(BenchJob(id=ident, **payload))
    return rows
