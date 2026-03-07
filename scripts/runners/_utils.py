import subprocess
import sys
from typing import List


def run_cmd(args: List[str]) -> None:
    printable = " ".join(args)
    print(f"[RUN] {printable}")
    subprocess.run(args, check=True)


def python_module_cmd(module: str, extra_args: List[str]) -> List[str]:
    return [sys.executable, "-m", module, *extra_args]
