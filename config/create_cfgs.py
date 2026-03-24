import itertools
from pathlib import Path
import yaml
import pandas as pd
import copy

# 1. Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_CFG_PATH = SCRIPT_DIR / "train.yaml"
OUT_DIR = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2) Sampling combos (100/50: with & without replacement; 10/1: with replacement only)
sampling_pairs = [
    (100, True), (100, False),
    (50,  True), (50,  False),
    (10,  True),
    (1,   True),
]

FORCED = {
    "data": {
        "force_aging_to_value": 0.9,   # <- your example
        # "truncate_noise": True,
    },
    "train": {
        "batch_size": 8192,
        "n_epochs": 2000,
    }
}

def apply_forced(cfg: dict, forced: dict) -> None:
    for sect, params in forced.items():
        cfg.setdefault(sect, {})
        cfg[sect].update(params)
        
def fmt(v):
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        s = f"{v}".rstrip("0").rstrip(".")
        return s.replace(".", "p") if s else "0"
    return str(v)

def forced_suffix(forced: dict) -> str:
    # _force_aging_to_value0p9_batch_size8192_n_epochs700
    tokens = []
    for _, params in forced.items():
        for k, v in params.items():
            tokens.append(f"{k}{fmt(v)}")
    return "_" + "_".join(tokens) if tokens else ""
    
# 3. Load base config
with BASE_CFG_PATH.open() as f:
    base_cfg = yaml.safe_load(f)

# 4. Store all config metadata
records = []
suffix = forced_suffix(FORCED)
i = 0
# 5. Iterate over all combinations
for sgs, swr in sampling_pairs:
    i += 1
    cfg = copy.deepcopy(base_cfg)
    # Ensure sections exist
    cfg.setdefault("data", {})
    cfg["data"]["sample_group_size"] = int(sgs)
    cfg["data"]["sample_with_replacement"] = bool(swr)
    
    # Apply forced overrides
    apply_forced(cfg, FORCED)

    
    fname = f"train-{i}_sgs{sgs}_repl{int(swr)}{suffix}.yaml"


    out_path = OUT_DIR / fname
    with out_path.open("w") as outf:
        yaml.safe_dump(cfg, outf, sort_keys=False)

    print(f"Written config: {out_path}")

    rec = {
        "filename": fname,
        "sample_group_size": sgs,
        "sample_with_replacement": swr,
    }
    # also log forced values into CSV as flattened columns
    for sect, params in FORCED.items():
        for k, v in params.items():
            rec[f"{sect}.{k}"] = v
    records.append(rec)


## ---- Index CSV ----
df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "config_index.csv", index=False)
print("Saved config index to config_index.csv")
