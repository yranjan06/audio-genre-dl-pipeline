import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Messy Mashup pipeline")
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="artifacts")
    parser.add_argument("--skip-hubert", action="store_true")
    parser.add_argument("--skip-crnn", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    return parser.parse_args()


def run(cmd):
    printable = " ".join(cmd)
    print(f"\n[RUN] {printable}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    generated = []

    if not args.skip_xgb:
        out = root / "xgb"
        run(
            [
                sys.executable,
                "-m",
                "scripts.train",
                "--base-dir",
                args.base_dir,
                "--model-type",
                "xgb",
                "--output-dir",
                str(out),
            ]
        )
        sub_csv = root / "submission_xgb.csv"
        run(
            [
                sys.executable,
                "-m",
                "scripts.predict",
                "--base-dir",
                args.base_dir,
                "--model-type",
                "xgb",
                "--checkpoint",
                str(out / "best_xgb.pkl"),
                "--output-csv",
                str(sub_csv),
            ]
        )
        generated.append(str(sub_csv))

    if not args.skip_cnn:
        out = root / "cnn"
        run(
            [
                sys.executable,
                "-m",
                "scripts.train",
                "--base-dir",
                args.base_dir,
                "--model-type",
                "cnn",
                "--epochs",
                "12",
                "--batch-size",
                "32",
                "--output-dir",
                str(out),
            ]
        )
        sub_csv = root / "submission_cnn.csv"
        run(
            [
                sys.executable,
                "-m",
                "scripts.predict",
                "--base-dir",
                args.base_dir,
                "--model-type",
                "cnn",
                "--checkpoint",
                str(out / "best_cnn.pth"),
                "--output-csv",
                str(sub_csv),
            ]
        )
        generated.append(str(sub_csv))

    if not args.skip_crnn:
        out = root / "crnn"
        run(
            [
                sys.executable,
                "-m",
                "scripts.train",
                "--base-dir",
                args.base_dir,
                "--model-type",
                "crnn",
                "--epochs",
                "12",
                "--batch-size",
                "24",
                "--output-dir",
                str(out),
            ]
        )
        sub_csv = root / "submission_crnn.csv"
        run(
            [
                sys.executable,
                "-m",
                "scripts.predict",
                "--base-dir",
                args.base_dir,
                "--model-type",
                "crnn",
                "--checkpoint",
                str(out / "best_crnn.pth"),
                "--output-csv",
                str(sub_csv),
            ]
        )
        generated.append(str(sub_csv))

    if not args.skip_hubert:
        out = root / "hubert"
        run(
            [
                sys.executable,
                "-m",
                "scripts.train",
                "--base-dir",
                args.base_dir,
                "--model-type",
                "hubert",
                "--epochs",
                "5",
                "--batch-size",
                "8",
                "--lr",
                "2e-5",
                "--freeze-feature-encoder",
                "--output-dir",
                str(out),
            ]
        )
        sub_csv = root / "submission_hubert.csv"
        run(
            [
                sys.executable,
                "-m",
                "scripts.predict",
                "--base-dir",
                args.base_dir,
                "--model-type",
                "hubert",
                "--checkpoint",
                str(out / "best_hubert.pth"),
                "--output-csv",
                str(sub_csv),
            ]
        )
        generated.append(str(sub_csv))

    if len(generated) >= 2:
        run(
            [
                sys.executable,
                "-m",
                "scripts.ensemble",
                "--inputs",
                *generated,
                "--output",
                str(root / "submission_ensemble.csv"),
            ]
        )

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
