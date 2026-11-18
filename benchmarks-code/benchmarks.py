from benchmarks.schedule import scheduler
import argparse
from benchmarks.initation import Config


if __name__ == "__main__":

    # Keep only the configuration path argument
    parser = argparse.ArgumentParser(description="Sequence Benchmarking Tools")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Configuration file path (default: config.yaml)")

    args = parser.parse_args()

    # Load the configuration file
    config = Config(args)

    # Start the scheduler
    scheduler(config)
