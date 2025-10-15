# IoT Intrusion Detection System
# Source code package initialization

from pathlib import Path


def resolve_config_path(default: str = "config.yaml") -> str:
    """Resolve config path preferring configs/config.yaml if present.

    Args:
        default: Fallback path

    Returns:
        Path string to config file
    """
    root = Path(__file__).parent.parent
    candidates = [
        root / "configs" / "config.yaml",
        root / default,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return default
