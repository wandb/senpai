import os
import sys
import wandb
import pandas as pd

api = wandb.Api()

entity = "wandb-applied-ai-team"
project = "senpai-v1"

run_ids = [
    ("thssa2ru", "Baseline s42",        {"p_in": 13.2, "p_tan": 31.4, "p_oodc": 7.7, "p_re": 6.3}),
    ("4lj8ch8s", "Baseline s43",        {"p_in": 13.3, "p_tan": 30.9, "p_oodc": 7.6, "p_re": 6.6}),
    ("qq24kqqv", "Distill a=0.6 s42",  {"p_in": 14.4, "p_tan": 31.6, "p_oodc": 8.0, "p_re": 6.6}),
    ("o2ir8mc6", "Distill a=0.6 s43",  {"p_in": 14.3, "p_tan": 32.2, "p_oodc": 8.3, "p_re": 6.8}),
    ("fjqz35fa", "Distill a=0.7 s42",  {"p_in": 14.1, "p_tan": 30.8, "p_oodc": 7.7, "p_re": 6.5}),
    ("m5smb3w6", "Distill a=0.7 s43",  {"p_in": 13.6, "p_tan": 30.3, "p_oodc": 8.0, "p_re": 6.5}),
    ("whoha5za", "Distill a=0.8 s42",  {"p_in": 13.2, "p_tan": 30.5, "p_oodc": 7.8, "p_re": 6.5}),
    ("j2pc23lk", "Distill a=0.8 s43",  {"p_in": 13.2, "p_tan": 30.4, "p_oodc": 7.8, "p_re": 6.6}),
]

metric_keys = ["surface_mae/p_in", "surface_mae/p_oodc", "surface_mae/p_tan", "surface_mae/p_re",
               "p_in", "p_oodc", "p_tan", "p_re",
               "val/p_in", "val/p_oodc", "val/p_tan", "val/p_re",
               "test/p_in", "test/p_oodc", "test/p_tan", "test/p_re",
               "_step", "epoch"]

rows = []
for run_id, label, student_reported in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        sm = run.summary_metrics

        # Print all summary keys for inspection on first run
        if run_id == "thssa2ru":
            print("=== Summary keys for thssa2ru ===")
            for k in sorted(sm.keys()):
                print(f"  {k}: {sm[k]}")
            print()

        epochs = sm.get("epoch", sm.get("_step", None))
        state = run.state

        # Try to find MAE metrics under various naming conventions
        def get_metric(sm, *candidates):
            for c in candidates:
                v = sm.get(c)
                if v is not None:
                    return round(float(v), 2)
            return None

        p_in   = get_metric(sm, "surface_mae/p_in",   "p_in",   "val/p_in",   "test/p_in")
        p_tan  = get_metric(sm, "surface_mae/p_tan",  "p_tan",  "val/p_tan",  "test/p_tan")
        p_oodc = get_metric(sm, "surface_mae/p_oodc", "p_oodc", "val/p_oodc", "test/p_oodc")
        p_re   = get_metric(sm, "surface_mae/p_re",   "p_re",   "val/p_re",   "test/p_re")

        rows.append({
            "run_id": run_id,
            "label": label,
            "state": state,
            "epochs": epochs,
            "actual_p_in":   p_in,
            "actual_p_tan":  p_tan,
            "actual_p_oodc": p_oodc,
            "actual_p_re":   p_re,
            "rep_p_in":   student_reported["p_in"],
            "rep_p_tan":  student_reported["p_tan"],
            "rep_p_oodc": student_reported["p_oodc"],
            "rep_p_re":   student_reported["p_re"],
        })
    except Exception as e:
        rows.append({"run_id": run_id, "label": label, "error": str(e)})

df = pd.DataFrame(rows)
print("=== Run Summary ===")
for _, r in df.iterrows():
    if "error" in r and pd.notna(r.get("error")):
        print(f"\n{r['run_id']} ({r['label']}): ERROR — {r['error']}")
        continue
    print(f"\n{r['run_id']} ({r['label']})")
    print(f"  State: {r['state']}  |  Epochs: {r['epochs']}")
    print(f"  Actual:   p_in={r['actual_p_in']}, p_tan={r['actual_p_tan']}, p_oodc={r['actual_p_oodc']}, p_re={r['actual_p_re']}")
    print(f"  Reported: p_in={r['rep_p_in']}, p_tan={r['rep_p_tan']}, p_oodc={r['rep_p_oodc']}, p_re={r['rep_p_re']}")

    discrepancies = []
    for metric in ["p_in", "p_tan", "p_oodc", "p_re"]:
        actual = r.get(f"actual_{metric}")
        reported = r.get(f"rep_{metric}")
        if actual is not None and reported is not None:
            diff = abs(actual - reported)
            if diff > 0.1:
                discrepancies.append(f"{metric}: actual={actual} vs reported={reported} (diff={diff:.2f})")
    if discrepancies:
        print(f"  DISCREPANCIES: {'; '.join(discrepancies)}")
    else:
        print(f"  No significant discrepancies (all within 0.1)")
