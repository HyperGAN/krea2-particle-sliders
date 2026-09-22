"""Sample-only entry. Same trainer, with ``--load_te_lora`` required."""

from __future__ import annotations

from krea2.train import build_parser, train


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    parser.description = (
        "Sample a Krea2 turbo-bbox slider from this repo. "
        "Uses the train entrypoint with --load_te_lora, which skips the "
        "train loop and writes the smile-first grid. No Hub download unless "
        "you pass --allow_hub on a live (non-dummy) run."
    )
    args = parser.parse_args(argv)
    if not args.load_te_lora and not args.print_card:
        parser.error(
            "--load_te_lora PATH is required. It skips training and writes "
            "the sample grid. Pass --dummy for the CPU stand-in."
        )
    if not args.dummy and not args.print_card:
        parser.error(
            "CPU infer needs --dummy. The live loader is krea2.live and is "
            "not started from this command, so Hub weights are not downloaded."
        )
    train(args)


if __name__ == "__main__":
    main()
