import wandb
import os

api = wandb.Api()
entity = "senpai-wandb"
project = "senpai"

# Try fetching runs by searching for them
run_ids = ["rqpfsjey", "v56f12x9"]

# First, try to list recent runs and find these IDs
print("Searching for runs by ID in recent runs...")
try:
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    count = 0
    found_ids = {}
    for run in runs:
        if count >= 300:
            break
        if run.id in run_ids:
            found_ids[run.id] = run
            print(f"  FOUND: {run.id} - name={run.name}, state={run.state}")
        count += 1
    print(f"  Scanned {count} runs, found {len(found_ids)} matches")
except Exception as e:
    print(f"  Error listing runs: {e}")

# Try alternate API access patterns
print("\nTrying alternate run access...")
for run_id in run_ids:
    # Try with just id
    try:
        runs_filtered = api.runs(
            f"{entity}/{project}",
            filters={"$or": [{"name": run_id}, {"run_id": run_id}]}
        )
        for r in runs_filtered[:5]:
            print(f"  Filter match: id={r.id}, name={r.name}")
    except Exception as e:
        print(f"  Filter error for {run_id}: {e}")

# Also try direct path with entity/project/runid format
print("\nTrying direct path access with different formats...")
for run_id in run_ids:
    for path_format in [
        f"{entity}/{project}/{run_id}",
        f"{project}/{run_id}",
    ]:
        try:
            run = api.run(path_format)
            print(f"  SUCCESS with path '{path_format}': name={run.name}")
            break
        except Exception as e:
            print(f"  FAIL with path '{path_format}': {e}")

print("\nDone.")
