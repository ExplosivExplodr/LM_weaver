"""Entry point for `python3 -m lm_weaver`.

Defaults to the trainer; use `python3 -m lm_weaver train` or `... chat`.
"""
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    command = argv[0] if argv else "train"

    if command in ("-h", "--help"):
        print(
            "usage: python3 -m lm_weaver <train|chat> [options]\n\n"
            "  train   train a model (see python3 -m lm_weaver train --help)\n"
            "  chat    chat with a checkpoint (see python3 -m lm_weaver chat --help)"
        )
        return 0

    if command == "train":
        from . import trainer as module
        sys.argv = ["trainer"] + argv[1:]
    elif command == "chat":
        from . import loader as module
        sys.argv = ["loader"] + argv[1:]
    else:
        print(f"Unknown command: {command!r}. Use 'train' or 'chat'.", file=sys.stderr)
        return 2

    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
