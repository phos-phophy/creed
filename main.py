import argparse
import json
from pathlib import Path

from src.manager import ManagerConfig, ModelManager


def load_manager():
    """ Parse arguments and load main config """

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, metavar="<path to config>", required=True)

    arguments = parser.parse_args()
    config_path = arguments.config_path

    with Path(config_path).open('r') as file:
        config: ManagerConfig = ManagerConfig.from_dict(json.load(file))

    return ModelManager(config)


def main():
    manager = load_manager()

    manager.train()
    manager.evaluate()
    manager.test()
    manager.predict()


if __name__ == '__main__':
    main()
