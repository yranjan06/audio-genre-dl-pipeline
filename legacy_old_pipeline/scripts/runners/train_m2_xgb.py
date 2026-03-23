import argparse
from scripts.runners._utils import python_module_cmd, run_cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 2: Train classical XGB model")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/m2_xgb")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = python_module_cmd(
        "scripts.train",
        [
            "--base-dir",
            args.base_dir,
            "--model-type",
            "xgb",
            "--output-dir",
            args.output_dir,
            *(["--no-wandb"] if args.no_wandb else []),
        ],
    )
    run_cmd(cmd)


if __name__ == "__main__":
    main()
