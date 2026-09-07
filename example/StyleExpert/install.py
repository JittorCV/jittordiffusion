#!/usr/bin/env python
"""Install StyleExpert's Jittor stack without the fragile torch stub URL."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def pip_install(*args: str) -> None:
    command = [PYTHON, "-m", "pip", "install", *args]
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command)


def main() -> None:
    pip_install("git+https://github.com/JittorRepos/jittor.git")
    # JTorch normally downloads torch metadata from pypi.jittor.org.  The local
    # metadata-only wheel below is equivalent and avoids that external SPOF.
    pip_install("--no-deps", "git+https://github.com/JittorRepos/jtorch.git")
    pip_install(str(ROOT / "compat" / "torch_stub"))
    pip_install("-r", str(ROOT / "requirements.txt"))


if __name__ == "__main__":
    main()

