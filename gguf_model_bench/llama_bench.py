from __future__ import annotations

from time import perf_counter
import json
import subprocess

from .planner import BenchJob


def command(job: BenchJob, binary: str = "llama-bench") -> list[str]:
    cmd = [
        binary,
        "-m", job.model,
        "-p", str(job.prompt_tokens),
        "-n", str(job.generation_tokens),
        "-d", str(job.depth),
        "-b", str(job.batch_size),
        "-ub", str(job.ubatch_size),
        "-ctk", job.cache_k,
        "-ctv", job.cache_v,
        "-ngl", str(job.gpu_layers),
        "-r", str(job.repetitions),
        "-o", "json",
    ]
    if job.threads is not None:
        cmd += ["-t", str(job.threads)]
    return cmd


def run(job: BenchJob, binary: str = "llama-bench") -> dict:
    cmd = command(job, binary)
    started = perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = perf_counter() - started

    parsed = None
    if proc.returncode == 0:
        parsed = json.loads(proc.stdout)

    return {
        "job_id": job.id,
        "job": job.to_dict(),
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "rows": parsed,
        "stderr": proc.stderr,
    }
