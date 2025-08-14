"""
Utilities for loading and handling configuration files.
"""

import yaml


def load_config(config_path: str = "config/config.yaml"):
    """
    Loads configuration settings from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file. Defaults to "config/config.yaml".
    Returns:
        dict: Configuration parameters loaded from the YAML file.
    Raises:
        FileNotFoundError: If the configuration file does not exist at the specified path.
        ValueError: If there is an error parsing the YAML file.
    """

    print(f"\n[INFO] Loading Configurations from {config_path}...")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from exc
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}") from e


def main():
    """
    This function serves as an entry point for the script.
    Prints a welcome message indicating that there is no action required in this script.
    """

    print("Welcome from config_utils.py. Nothing to Do ^_____^")


if __name__ == "__main__":
    main()