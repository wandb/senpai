import pandas as pd
import numpy as np

data = [
    {"condition": "control", "seed": 42, "run_id": "wwil2gdr", "p_in": 12.479847, "p_oodc": 7.407363, "p_tan": 30.767057, "p_re": 6.492333},
    {"condition": "control", "seed": 43, "run_id": "bfa65aup", "p_in": 13.242900, "p_oodc": 7.939553, "p_tan": 30.906664, "p_re": 6.477128},
    {"condition": "K=4",     "seed": 42, "run_id": "z3b8tdfy", "p_in": 12.800614, "p_oodc": 7.772099, "p_tan": 30.058634, "p_re": 6.545139},
    {"condition": "K=4",     "seed": 43, "run_id": "p5pljk4j", "p_in": 13.225006, "p_oodc": 7.699864, "p_tan": 29.358376, "p_re": 6.303254},
    {"condition": "K=8",     "seed": 42, "run_id": "nicjx1g0", "p_in": 13.318239, "p_oodc": 7.675220, "p_tan": 29.303448, "p_re": 6.455322},
    {"condition": "K=8",     "seed": 43, "run_id": "a4jaduno", "p_in": 13.316330, "p_oodc": 7.939188, "p_tan": 31.550035, "p_re": 6.687196},
]
df = pd.DataFrame(data)

means = df.groupby("condition")[["p_in", "p_oodc", "p_tan", "p_re"]].mean()
means.index = pd.CategoricalIndex(means.index, categories=["control", "K=4", "K=8"], ordered=True)
means = means.sort_index()

print("=== 2-SEED MEANS BY CONDITION ===")
print(means.round(6).to_string())

print("\n=== VS CONTROL (delta, lower is better) ===")
ctrl = means.loc["control"]
for cond in ["K=4", "K=8"]:
    delta = means.loc[cond] - ctrl
    print(f"\n{cond} - control:")
    for col in ["p_in", "p_oodc", "p_tan", "p_re"]:
        sign = "+" if delta[col] > 0 else ""
        print(f"  {col}: {sign}{delta[col]:.6f}")
