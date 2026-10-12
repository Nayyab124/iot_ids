
"""Entry point for running the project."""

from iot_ids import __version__
from iot_ids.config import load_config


def main() -> None:
    config = load_config()
    project = config.get("project", {})

    print(f"{project.get('name', 'IoT IDS/IPS')} v{__version__}")
    print(f"Stage: {project.get('stage', 'not specified')}")
    print("Status: Phase 1 setup is working.")
    print("Network capture and firewall enforcement are disabled.")


if __name__ == "__main__":
    main()
