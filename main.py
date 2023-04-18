import argparse
import json
from pathlib import Path

from src.manager import ManagerConfig, ModelManager


def load_manager():
    """ Parses arguments and loads main config """

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, metavar='<path to config>', required=True)
    parser.add_argument('-s', '--seed', type=int, metavar='<seed>', required=True)
    parser.add_argument('-o', '--output_dir', type=str, metavar='<path to model output directory>', required=True)

    arguments = parser.parse_args()
    config_path = arguments.config_path
    seed = arguments.seed
    output_dir = arguments.output_dir

    with Path(config_path).open('r') as file:
        config_dict = json.load(file)
        config_dict['seed'] = seed
        config: ManagerConfig = ManagerConfig.from_dict(config_dict, output_dir)

    config_path = Path(output_dir) / "config.json"
    with config_path.open('w') as file:
        json.dump(config_dict, file)

    return ModelManager(config)


def main():
    manager = load_manager()

    manager.train()
    manager.evaluate()
    manager.test()
    manager.predict()


if __name__ == '__main__':
    main()
