#!/usr/bin/env python3
import csv
import os
import sqlite3
import sys
import time
from pathlib import Path


def option(name):
    prefix = f"--{name}="
    return next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith(prefix))


def write_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


with open(os.environ["FAKE_MSPROF_LOG"], "a", encoding="utf-8") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")

is_op = len(sys.argv) > 1 and sys.argv[1] == "op"
output = Path(option("output"))
sleep_seconds = float(os.environ.get("FAKE_MSPROF_SLEEP_SEC", "0"))
if sleep_seconds:
    time.sleep(sleep_seconds)
if is_op:
    if os.environ.get("FAKE_MSPROF_FAIL_OP") == "1":
        raise SystemExit(9)
    op_dir = output / "OPPROF_fake" / "add_kernel_hash" / "0"
    write_csv(
        op_dir / "OpBasicInfo.csv",
        {
            "Op Name": "add_kernel_hash",
            "Op Type": "vector",
            "Task Duration(us)": "6",
            "Block Dim": "8",
        },
    )
    write_csv(
        op_dir / "PipeUtilization.csv",
        {
            "aiv_vec_ratio": "0.6",
            "aiv_mte2_ratio": "0.3",
            "aiv_mte3_ratio": "0.1",
        },
    )
    write_csv(
        op_dir / "Memory.csv",
        {
            "GM_to_UB_bw_usage_rate(%)": "42",
            "UB_to_GM_bw_usage_rate(%)": "21",
        },
    )
    write_csv(op_dir / "L2Cache.csv", {"aiv_total_hit_rate(%)": "90"})
    raise SystemExit(0)

metric = option("aic-metrics")
if "--aic-mode=sample-based" in sys.argv:
    if os.environ.get("FAKE_MSPROF_FAIL_SAMPLE") == "1":
        raise SystemExit(8)
    db_dir = output / "PROF_fake" / "device_0" / "sqlite"
    db_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_dir / "aicore.db")
    connection.execute(
        "CREATE TABLE AICoreOriginalData(coreid INTEGER, task_cyc INTEGER)"
    )
    connection.executemany(
        "INSERT INTO AICoreOriginalData VALUES (?, ?)",
        [(0, 100), (1, 90)],
    )
    connection.commit()
    connection.close()
    raise SystemExit(0)

if os.environ.get("FAKE_MSPROF_FAIL_METRIC") == metric:
    raise SystemExit(7)

row = {
    "Op Name": "add_kernel_hash",
    "Task Duration(us)": "6",
    "aicore_time(us)": "10",
}
if metric == "PipeUtilization":
    row.update(
        {
            "aiv_mte2_ratio": "0.85",
            "aic_cube_ratio": "0.10",
            "pushq_ratio": "0.10",
            "aic_fixpipe_ratio": "0.10",
            "aiv_mte3_ratio": "0.10",
            "aiv_scalar_ratio": "0.10",
            "scalarldst_ratio": "0.10",
        }
    )
write_csv(
    output / "PROF_fake" / "mindstudio_profiler_output" / "op_summary_0.csv",
    row,
)
