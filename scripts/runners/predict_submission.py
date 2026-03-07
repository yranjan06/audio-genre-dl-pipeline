import argparse
from scripts.runners._utils import python_module_cmd, run_cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and write submission CSV")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--model-type", choices=["xgb", "cnn", "crnn", "hubert"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", default="submissions/submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = python_module_cmd(
        "scripts.predict",
        [
            "--base-dir",
            args.base_dir,
            "--model-type",
            args.model_type,
            "--checkpoint",
            args.checkpoint,
            "--output-csv",
            args.output_csv,
        ],
    )
    run_cmd(cmd)


if __name__ == "__main__":
    main()
