"""Console entry point: ``python -m eval --config <path>``."""

from __future__ import annotations

from .cli import parse_args
from .run import run


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
