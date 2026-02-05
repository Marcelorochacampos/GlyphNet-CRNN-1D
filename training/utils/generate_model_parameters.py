import json
import yaml
import os
import itertools
import hashlib
from copy import deepcopy


CHECKPOINT_PATH = "./checkpoint/model_progress_checkpoint.json"


BASE_CHECKPOINT = {
    "amount_of_models_to_train": 16,
    "models": []
}


def param_hash(params: dict) -> str:
    normalized = json.dumps(params, sort_keys=True)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def expand_grid(grid: dict):
    keys = []
    values = []

    for section, params in grid.items():
        if not isinstance(params, dict):
            continue

        for key, value in params.items():
            if isinstance(value, list):
                keys.append((section, key))
                values.append(value)

    for combination in itertools.product(*values):
        cfg = {}
        for (section, key), val in zip(keys, combination):
            cfg.setdefault(section, {})
            cfg[section][key] = val
        yield cfg


def merge_configs(defaults: dict, overrides: dict) -> dict:
    result = deepcopy(defaults)
    for section, params in overrides.items():
        for key, value in params.items():
            result[section][key] = value
    return result


def generate_model_parameters():
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    if not os.path.exists(CHECKPOINT_PATH):
        print("[INFO] Checkpoint not found, creating a new one.")
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(BASE_CHECKPOINT, f, indent=4)

    with open(CHECKPOINT_PATH, "r") as f:
        progress_checkpoint = json.load(f)

    amount_of_models_to_train = progress_checkpoint["amount_of_models_to_train"]
    existing_models = progress_checkpoint["models"]

    existing_hashes = {
        model["param_hash"] for model in existing_models
    }

    remaining = amount_of_models_to_train - len(existing_models)
    print(f"[INFO] Remaining models to generate: {remaining}")

    if remaining <= 0:
        return progress_checkpoint

    with open("./config/default_parameters.yaml", "r") as f:
        default_parameters = yaml.safe_load(f)

    with open("./config/grid_search_parameters.yaml", "r") as f:
        grid_search_parameters = yaml.safe_load(f)

    generated = 0

    for grid_override in expand_grid(grid_search_parameters):
        full_config = merge_configs(default_parameters, grid_override)
        h = param_hash(full_config)

        if h in existing_hashes:
            continue

        model_id = f"model_{len(existing_models):04d}"

        existing_models.append({
            "id": model_id,
            "param_hash": h,
            "parameters": full_config,
            "trained": False,
            "val_loss": None,
            "model_path": None
        })

        existing_hashes.add(h)
        generated += 1

        if generated >= remaining:
            break

    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(progress_checkpoint, f, indent=4)

    print(f"[INFO] Generated {generated} new model parameter sets.")
    return progress_checkpoint
