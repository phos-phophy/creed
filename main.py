import argparse
import json
from pathlib import Path

from src.manager import ManagerConfig, ModelManager


def load_manager():
    """ Parses arguments and loads main config """

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, metavar="<path to config>", required=True)
    parser.add_argument("-s", "--seed", type=int, metavar="<seed>", required=True)

    arguments = parser.parse_args()
    config_path = arguments.config_path
    seed = arguments.seed

    with Path(config_path).open('r') as file:
        config_dict = json.load(file)
        config_dict['seed'] = seed
        config: ManagerConfig = ManagerConfig.from_dict(config_dict)

    return ModelManager(config)


def main():
    manager = load_manager()

    manager.train()
    manager.evaluate()
    manager.test()
    manager.predict()


if __name__ == '__main__':
    main()
