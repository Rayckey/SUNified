"""Inspect the SUN Program skill grammar without a simulator."""

import argparse
import json

from .schema import OP_ARGS, PROGRAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=sorted(OP_ARGS))
    args = parser.parse_args()
    print(json.dumps(OP_ARGS[args.operator] if args.operator else PROGRAM, indent=2))


if __name__ == "__main__":
    main()
