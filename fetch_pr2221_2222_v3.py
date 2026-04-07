import os
import wandb

api = wandb.Api()
entity = "senpai-wandb"

# List available projects
try:
    projects = api.projects(entity)
    print("Projects:")
    for p in projects:
        print(f"  {p.name}")
except Exception as e:
    print(f"Error listing projects: {e}")
