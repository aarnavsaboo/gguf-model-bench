from collections import defaultdict
from statistics import median


def flatten(run_records: list[dict]) -> list[dict]:
    rows = []
    for record in run_records:
        if record.get("returncode") != 0 or not record.get("rows"):
            continue
        for bench in record["rows"]:
            rows.append({
                "job_id": record["job_id"],
                "model": record["job"]["model"],
                **bench,
            })
    return rows


def summarize(run_records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in flatten(run_records):
        test = row.get("test", "")
        key = (row["model"], test, row.get("n_prompt"), row.get("n_gen"), row.get("n_depth"))
        groups[key].append(row)

    output = []
    for key, group in sorted(groups.items(), key=lambda x: str(x[0])):
        model, test, n_prompt, n_gen, n_depth = key
        rates = [float(x["avg_ts"]) for x in group if x.get("avg_ts") is not None]
        output.append({
            "model": model,
            "test": test,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "n_depth": n_depth,
            "rows": len(group),
            "median_tokens_per_second": None if not rates else median(rates),
            "min_tokens_per_second": None if not rates else min(rates),
            "max_tokens_per_second": None if not rates else max(rates),
        })
    return output
