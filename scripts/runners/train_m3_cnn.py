import argparse
from scripts.runners._utils import python_module_cmd, run_cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 3: Train CNN model")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/m3_cnn")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
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
            "cnn",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--output-dir",
            args.output_dir,
            *(["--no-wandb"] if args.no_wandb else []),
        ],
    )
    run_cmd(cmd)


if __name__ == "__main__":
    main()
