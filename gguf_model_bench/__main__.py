from argparse import ArgumentParser
from pathlib import Path
import json

from .catalog import discover
from .io import read_jobs, read_rows, write_rows
from .llama_bench import run
from .planner import expand
from .report import summarize


def main():
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("path")

    plan = sub.add_parser("plan")
    plan.add_argument("config")

    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("plan")
    run_cmd.add_argument("--out", required=True)
    run_cmd.add_argument("--binary", default="llama-bench")

    report = sub.add_parser("report")
    report.add_argument("path")

    args = parser.parse_args()

    if args.cmd == "inspect":
        for row in discover(args.path):
            print(json.dumps(row, sort_keys=True))
    elif args.cmd == "plan":
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for job in expand(config):
            print(json.dumps(job.to_dict(), sort_keys=True))
    elif args.cmd == "run":
        rows = [run(job, args.binary) for job in read_jobs(args.plan)]
        write_rows(args.out, rows)
        print(json.dumps({"jobs": len(rows), "successful": sum(x["returncode"] == 0 for x in rows)}))
    else:
        print(json.dumps(summarize(read_rows(args.path)), indent=2))


if __name__ == "__main__":
    main()
