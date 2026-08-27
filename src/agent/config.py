"""Load model/agent config (YAML) and the API key (.env, never committed)."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")


def load_yaml(relative_path):
    with (REPO_ROOT / relative_path).open() as f:
        return yaml.safe_load(f)


def load_model_config():
    config = load_yaml("configs/models.yaml")
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"{config['api_key_env']} is not set. Add it to .env or export it before running."
        )
    config["api_key"] = api_key
    return config


def load_agent_config():
    return load_yaml("configs/agent.yaml")


def load_experiment_config(name):
    return load_yaml("configs/experiments.yaml")[name]
