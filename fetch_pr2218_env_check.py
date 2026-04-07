import wandb
import os

# Check env vars
print("WANDB_ENTITY:", os.environ.get("WANDB_ENTITY", "NOT SET"))
print("WANDB_PROJECT:", os.environ.get("WANDB_PROJECT", "NOT SET"))
print("WANDB_API_KEY set:", "YES" if os.environ.get("WANDB_API_KEY") else "NO")

api = wandb.Api()

# Check who we are logged in as
try:
    viewer = api.viewer
    print(f"\nLogged in as: {viewer.entity}")
    print(f"Teams: {[t for t in viewer.teams]}")
except Exception as e:
    print(f"Viewer error: {e}")

# Try to list projects for the entity
try:
    projects = api.projects("senpai-wandb")
    print(f"\nProjects under senpai-wandb:")
    for p in projects:
        print(f"  {p.name}")
except Exception as e:
    print(f"Projects error for senpai-wandb: {e}")

# Try default entity
try:
    entity = os.environ.get("WANDB_ENTITY", "")
    if entity:
        projects = api.projects(entity)
        print(f"\nProjects under {entity}:")
        for p in projects:
            print(f"  {p.name}")
except Exception as e:
    print(f"Projects error: {e}")

print("\nDone.")
