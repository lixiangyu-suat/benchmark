import yaml


def load_config(path):
    """Load a YAML config file and return a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

