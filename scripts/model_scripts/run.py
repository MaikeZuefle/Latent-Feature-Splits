import argparse
import os

import yaml
from model import run_cl

parser = argparse.ArgumentParser()
parser.add_argument(
    "yaml_path", metavar="path", type=str, help="path to yaml file (embeddings)"
)
parser.add_argument(
    "test",
    help="only test the model without training",
    nargs="?",
    type=bool,
    default=False,
)
args = parser.parse_args()


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def main():
    """Main method which reads a yaml file and starts the training of the model with the specified parameters.
    """
    with open(args.yaml_path, encoding="UTF8") as y:
        conf = yaml.load(y.read(), Loader=yaml.FullLoader)
        only_test = args.test
        run_cl(**conf["classifier"], only_test=only_test)


if __name__ == "__main__":
    main()
