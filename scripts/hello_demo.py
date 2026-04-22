from __future__ import annotations

import sys
from datetime import datetime


def main() -> None:
    print("Hello from XPU Agent!")
    print(f"Current datetime: {datetime.now()}")
    print(f"Python version: {sys.version}")


if __name__ == "__main__":
    main()
