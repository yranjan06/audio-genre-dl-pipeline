import argparse
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Majority-vote ensemble for submission CSV files")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input submission CSV files with columns: id,genre",
    )
    parser.add_argument("--output", type=str, default="submission_ensemble.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dfs: List[pd.DataFrame] = [pd.read_csv(path, dtype={"id": str}) for path in args.inputs]

    base_ids = dfs[0]["id"].tolist()
    for idx, df in enumerate(dfs[1:], start=1):
        if df["id"].tolist() != base_ids:
            raise ValueError(f"ID order mismatch in file index={idx}: {args.inputs[idx]}")

    rows = []
    for i, file_id in enumerate(base_ids):
        genres = [df.iloc[i]["genre"] for df in dfs]
        voted = Counter(genres).most_common(1)[0][0]
        rows.append({"id": file_id, "genre": voted})

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved ensemble submission: {out_path}")


if __name__ == "__main__":
    main()
